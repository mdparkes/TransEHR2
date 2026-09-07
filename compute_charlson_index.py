#!/usr/bin/env python3
"""Score each episode's most recent earlier hospital admission with the Charlson index.

The extracted arrays carry pre-admission discharge diagnoses only as embedded text, so the
index has to be computed from the per-subject CSVs the extraction was built from and joined
back onto the arrays afterwards. This script does the computation once and writes a table
keyed by patient-episode ID, which `run_charlson_logistic_regression.py` then joins.

Which admission is scored
-------------------------
`stays.csv` holds one row per retained hospital admission, sorted by ICU `INTIME`, and episode
number `n` is its `n`-th row. The extraction emits one diagnosis-descriptions record per row
of that file, timestamped at the row's `DISCHTIME`, and blanks any whose timestamp falls after
the current episode's `INTIME`. The records an episode can see are therefore exactly the rows
whose `DISCHTIME` is at or before its own `INTIME`, and the most recent set of discharge
diagnoses is the latest `DISCHTIME` among them. That row's `HADM_ID` selects the codes from
`diagnoses.csv`.

Reproducing the extraction's rule here rather than re-deriving one is the point: the cohort is
decided from the arrays, so any disagreement about which records count would put the index and
the models on different episodes. `--check_cohort` verifies the agreement against a fold's
extracted arrays and reports every episode the two disagree on.

Outputs
-------
`charlson_index.csv`, one row per episode that has a scorable earlier admission:

    episode_id, patient_id, episode_number, charlson_index, n_conditions,
    source_hadm_id, n_codes, n_unmapped_codes, hours_before_admission,
    and one 0/1 column per comorbidity, named as in `TransEHR2.data.charlson`

Usage:
    python compute_charlson_index.py TransEHR2/configs/datasets/mimic4.yaml
    python compute_charlson_index.py TransEHR2/configs/datasets/mimic4.yaml -w 8
    python compute_charlson_index.py TransEHR2/configs/datasets/mimic4.yaml \
        --check_cohort fold1
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from TransEHR2.data.charlson import (CONDITION_KEYS, charlson_conditions,
                                     conditions_for_code, WEIGHTS)
from TransEHR2.data.cohorts import cohort_mask

DEFAULT_OUTPUT = os.path.join('misc', 'charlson', 'charlson_index.csv')
COHORT = 'diagnosis_history'


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def discover_folds(data_dir, requested_folds=None):
    """Discover fold directories, matching extract_data.py's pattern."""
    if requested_folds:
        return requested_folds
    return sorted([
        item for item in os.listdir(data_dir)
        if re.match(r'fold\d+', item) and os.path.isdir(os.path.join(data_dir, item))
    ])


def collect_episodes(data_dir, fold_names):
    """Collect every episode named in the fold listfiles.

    Folds partition the same episodes, so one episode is generally named in several folds.
    Scoring is a property of the episode and not of the fold, so they are deduplicated here
    and scored once.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold_names: Fold directory names to read.

    Returns:
        Dict mapping patient-episode ID to (patient_dir, episode_number).

    Raises:
        FileNotFoundError: If no listfile could be read at all.
    """
    episodes = {}
    n_read = 0
    for fold_name in fold_names:
        fold_dir = os.path.join(data_dir, fold_name)
        for partition in ('train', 'val', 'test'):
            path = os.path.join(fold_dir, f'{fold_name}_{partition}.csv')
            if not os.path.exists(path):
                continue
            frame = pd.read_csv(path)
            n_read += 1
            for episode_path, patient_id, episode_number in zip(
                    frame.iloc[:, 0], frame.iloc[:, 1].astype(int),
                    frame.iloc[:, 2].astype(int)):
                patient_dir = os.path.dirname(os.path.abspath(str(episode_path)))
                episode_id = int(patient_id) * 1000 + int(episode_number)
                episodes[episode_id] = (patient_dir, int(episode_number))

    if not n_read:
        raise FileNotFoundError(
            f'No {{fold}}_{{partition}}.csv listfiles found under {data_dir}. The episode '
            f'listfiles name the episodes to score; without them there is nothing to do.'
        )
    return episodes


# ---------------------------------------------------------------------------
# Per-patient scoring
# ---------------------------------------------------------------------------

def most_recent_earlier_admission(stays, episode_number):
    """The row of `stays.csv` holding the diagnoses an episode can see.

    Args:
        stays: `stays.csv` for one patient, sorted by INTIME, with INTIME and DISCHTIME
            parsed to datetimes.
        episode_number: 1-based episode number within that file.

    Returns:
        Tuple of (row, hours_before_admission), or (None, None) when the episode has no
        earlier admission whose discharge precedes its own ICU admission.

    Raises:
        IndexError: If `episode_number` is not a row of `stays`.
    """
    current = stays.iloc[episode_number - 1]
    intime = current['INTIME']

    # The extraction blanks a diagnosis record whose timestamp is after INTIME, so a record
    # survives iff its DISCHTIME <= INTIME. Excluding the current admission by HADM_ID as well
    # guards the boundary case of an admission discharged at the very moment of ICU admission.
    earlier = stays[(stays['DISCHTIME'] <= intime)
                    & (stays['HADM_ID'] != current['HADM_ID'])]
    earlier = earlier.dropna(subset=['DISCHTIME'])
    if earlier.empty:
        return None, None

    row = earlier.loc[earlier['DISCHTIME'].idxmax()]
    hours = (intime - row['DISCHTIME']).total_seconds() / 3600.0
    return row, float(hours)


def score_patient(task):
    """Score every requested episode of one patient.

    Args:
        task: Tuple of (patient_dir, [(episode_id, episode_number), ...]).

    Returns:
        List of per-episode result dicts. An episode with no scorable earlier admission is
        returned with `status` set and no index, so the caller can account for it rather than
        having it silently vanish.
    """
    patient_dir, records = task
    stays_path = os.path.join(patient_dir, 'stays.csv')
    diagnoses_path = os.path.join(patient_dir, 'diagnoses.csv')

    try:
        stays = pd.read_csv(stays_path)
    except FileNotFoundError:
        return [{'episode_id': eid, 'status': 'missing_stays'} for eid, _ in records]

    for column in ('INTIME', 'DISCHTIME'):
        if column not in stays.columns:
            return [{'episode_id': eid, 'status': 'missing_stay_times'} for eid, _ in records]
        stays[column] = pd.to_datetime(stays[column], errors='coerce')
    stays = stays.sort_values('INTIME').reset_index(drop=True)

    if os.path.exists(diagnoses_path):
        diagnoses = pd.read_csv(
            diagnoses_path, dtype={'ICD_CODE': str, 'ICD_VERSION': 'Int64'}
        )
    else:
        diagnoses = None

    results = []
    for episode_id, episode_number in records:
        if episode_number < 1 or episode_number > len(stays):
            results.append({'episode_id': episode_id, 'status': 'episode_out_of_range'})
            continue

        row, hours = most_recent_earlier_admission(stays, episode_number)
        if row is None:
            results.append({'episode_id': episode_id, 'status': 'no_earlier_admission'})
            continue
        if diagnoses is None:
            results.append({'episode_id': episode_id, 'status': 'missing_diagnoses'})
            continue

        source = diagnoses[diagnoses['HADM_ID'] == row['HADM_ID']]
        codes = [
            (code, version)
            for code, version in zip(source['ICD_CODE'], source['ICD_VERSION'])
            if pd.notna(code) and pd.notna(version)
        ]
        if not codes:
            results.append({'episode_id': episode_id, 'status': 'no_codes',
                            'source_hadm_id': int(row['HADM_ID']),
                            'hours_before_admission': hours})
            continue

        present = charlson_conditions(codes)
        n_unmapped = sum(1 for code, version in codes
                         if not conditions_for_code(code, version))
        result = {
            'episode_id': episode_id,
            'status': 'ok',
            'charlson_index': sum(WEIGHTS[key] for key, flag in present.items() if flag),
            'n_conditions': sum(1 for flag in present.values() if flag),
            'source_hadm_id': int(row['HADM_ID']),
            'n_codes': len(codes),
            'n_unmapped_codes': n_unmapped,
            'hours_before_admission': hours,
        }
        result.update({key: int(present[key]) for key in CONDITION_KEYS})
        results.append(result)

    return results


def score_all(episodes, n_workers):
    """Score every episode, grouping the work by patient.

    Args:
        episodes: Output of `collect_episodes`.
        n_workers: Worker processes; 1 runs in-process.

    Returns:
        List of per-episode result dicts.
    """
    by_patient = defaultdict(list)
    for episode_id, (patient_dir, episode_number) in episodes.items():
        by_patient[patient_dir].append((episode_id, episode_number))

    tasks = [(patient_dir, sorted(records))
             for patient_dir, records in sorted(by_patient.items())]

    results = []
    if n_workers <= 1:
        for i, task in enumerate(tasks, 1):
            results.extend(score_patient(task))
            if i % 500 == 0:
                print(f'    {i}/{len(tasks)} patients', flush=True)
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(score_patient, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            results.extend(future.result())
            if i % 500 == 0:
                print(f'    {i}/{len(tasks)} patients', flush=True)
    return results


# ---------------------------------------------------------------------------
# Cohort agreement
# ---------------------------------------------------------------------------

def check_cohort(data_dir, fold_name, scored, extracted_history_len_steps=None):
    """Compare the scored episodes against a fold's extracted cohort membership.

    The cohort is decided from the arrays and the index is computed from the CSVs, so the two
    must agree episode for episode or the index and the models would be reported on different
    populations. This reports the disagreement in both directions.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold_name: Fold whose partitions to check.
        scored: Set of patient-episode IDs that were scored successfully.
        extracted_history_len_steps: Width of the history region, for datasets written before
            the layout was recorded in metadata.

    Returns:
        Tuple of (n_in_cohort, missing, extra) where `missing` are cohort episodes with no
        index and `extra` are scored episodes outside the cohort, both as sorted ID lists.
    """
    from TransEHR2.data.preprocessing import load_dataset

    in_cohort = set()
    for split in ('train', 'val', 'test'):
        base = os.path.join(data_dir, fold_name, split)
        if not os.path.exists(os.path.join(base, 'metadata.pkl')):
            continue
        dataset = load_dataset(base, extracted_history_len_steps=extracted_history_len_steps)
        mask = cohort_mask(dataset, COHORT)

        ids_path = os.path.join(data_dir, fold_name, f'{split}_ids.pkl')
        with open(ids_path, 'rb') as handle:
            episode_ids = pickle.load(handle)
        if len(episode_ids) != len(mask):
            raise ValueError(
                f'{fold_name}/{split}: {len(mask)} episodes in the arrays but '
                f'{len(episode_ids)} ids in {ids_path}. The ids and the extracted arrays are '
                f'out of step, so cohort membership cannot be attributed to an episode.'
            )
        in_cohort.update(np.asarray(episode_ids, dtype=np.int64)[mask].tolist())

    if not in_cohort:
        raise FileNotFoundError(
            f'No extracted partitions found under {os.path.join(data_dir, fold_name)}, so '
            f'there is no cohort to check against.'
        )
    return len(in_cohort), sorted(in_cohort - scored), sorted(scored - in_cohort)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_table(results):
    """Assemble the output table from the per-episode results.

    Args:
        results: Output of `score_all`.

    Returns:
        Tuple of (frame, status_counts) where `frame` holds the scored episodes in ID order
        and `status_counts` counts every status seen, scored or not.
    """
    status_counts = defaultdict(int)
    for result in results:
        status_counts[result['status']] += 1

    scored = [result for result in results if result['status'] == 'ok']
    columns = ['episode_id', 'patient_id', 'episode_number', 'charlson_index', 'n_conditions',
               'source_hadm_id', 'n_codes', 'n_unmapped_codes', 'hours_before_admission',
               *CONDITION_KEYS]
    if not scored:
        return pd.DataFrame(columns=columns), dict(status_counts)

    frame = pd.DataFrame(scored).drop(columns=['status'])
    frame['patient_id'] = frame['episode_id'] // 1000
    frame['episode_number'] = frame['episode_id'] % 1000
    return frame[columns].sort_values('episode_id').reset_index(drop=True), dict(status_counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dataset_config', help='YAML file specifying dataset parameters')
    parser.add_argument('--data_dir', default=None,
                        help='Override DATA_DIR from the dataset config')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output CSV path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--folds', nargs='+', default=None,
                        help='Folds whose listfiles name the episodes to score '
                             '(default: every fold)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Worker processes (default: 1)')
    parser.add_argument('--check_cohort', metavar='FOLD', default=None,
                        help="Verify the scored episodes against this fold's extracted cohort "
                             'membership and exit non-zero on disagreement')
    args = parser.parse_args(argv)

    with open(args.dataset_config) as handle:
        dataset_config = yaml.safe_load(handle)
    data_dir = args.data_dir or dataset_config['DATA_DIR']

    fold_names = discover_folds(data_dir, args.folds)
    if not fold_names:
        raise SystemExit(f'No fold directories found in {data_dir}')
    print(f'Folds: {", ".join(fold_names)}')

    episodes = collect_episodes(data_dir, fold_names)
    print(f'Episodes named in the listfiles: {len(episodes)}')

    print('Scoring...')
    results = score_all(episodes, args.workers)
    frame, status_counts = build_table(results)

    print('\nStatus:')
    for status in sorted(status_counts):
        print(f'  {status:22} {status_counts[status]}')

    if not frame.empty:
        index = frame['charlson_index']
        print(f'\nCharlson index over {len(frame)} scored episodes: '
              f'mean {index.mean():.2f}, median {index.median():.0f}, '
              f'range {index.min()}-{index.max()}')
        print(f'Zero-index episodes: {int((index == 0).sum())} '
              f'({100.0 * (index == 0).mean():.1f}%)')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f'\nWrote {len(frame)} rows to {args.output}')

    if args.check_cohort:
        print(f'\nChecking cohort agreement on {args.check_cohort}...')
        n_in_cohort, missing, extra = check_cohort(
            data_dir, args.check_cohort, set(frame['episode_id'].tolist()),
            dataset_config.get('MAX_HISTORY_LEN_STEPS'),
        )
        print(f'  episodes in the {COHORT!r} cohort: {n_in_cohort}')
        print(f'  in the cohort with no index:      {len(missing)}')
        print(f'  scored but outside the cohort:    {len(extra)}')
        if missing or extra:
            for label, ids in (('no index', missing), ('outside cohort', extra)):
                if ids:
                    shown = ', '.join(str(i) for i in ids[:20])
                    more = '' if len(ids) <= 20 else f', ... ({len(ids)} total)'
                    print(f'  {label}: {shown}{more}', file=sys.stderr)
            print(
                '\nThe cohort is decided from the extracted arrays and the index from the '
                'per-subject CSVs. A disagreement means the two disagree about which earlier '
                'admissions an episode can see, so the index and the models would be reported '
                'on different populations.',
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
