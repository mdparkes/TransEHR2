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

The cohort
----------
`--write_cohort` also writes the episode manifest the analysis runs on: the episodes for which
all three features exist -- a Charlson index, an age and a sex. Both arms are handed that one
file, the regression directly and the in-stay-only control through `COHORT_EPISODES` in its
experiment config, and both resolve it by patient-episode ID through the same loader. That is
what puts them on the same episodes; nothing has to be argued about two predicates agreeing.

Membership is deliberately *not* the array-side predicate `diagnosis_history`, which asks
whether the text of an earlier admission's diagnoses survived extraction. Neither arm reads
that text -- the control reads in-stay records only, and the index comes from `diagnoses.csv`
-- and it excludes episodes whose codes are perfectly available. The printed funnel shows how
many episodes each condition removes.

Outputs
-------
`charlson_index.csv`, one row per episode that has a scorable earlier admission:

    episode_id, patient_id, episode_number, charlson_index, n_conditions,
    source_hadm_id, n_codes, n_unmapped_codes, hours_before_admission,
    and one 0/1 column per comorbidity, named as in `TransEHR2.data.charlson`

`charlson_cohort.txt` with `--write_cohort`, one patient-episode ID per line.

Usage:
    python compute_charlson_index.py TransEHR2/configs/datasets/mimic4.yaml -w 8 \
        --write_cohort
    python compute_charlson_index.py TransEHR2/configs/datasets/mimic4.yaml
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from TransEHR2.data.charlson import (CONDITION_KEYS, charlson_conditions,
                                     conditions_for_code, WEIGHTS)
from TransEHR2.data.statics import (MISSING_LABEL, age_observed, decode_categorical,
                                    static_offsets)

DEFAULT_OUTPUT = os.path.join('misc', 'charlson', 'charlson_index.csv')
DEFAULT_COHORT = os.path.join('misc', 'charlson', 'charlson_cohort.txt')


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
# The cohort manifest
# ---------------------------------------------------------------------------

def feature_availability(data_dir, fold, offsets, category_map,
                         extracted_history_len_steps=None):
    """Which extracted episodes of one fold have an age and a sex on record.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold whose partitions to read. Folds partition the same episodes, so one fold's
            three partitions cover the extraction once.
        offsets: Output of `static_offsets`.
        category_map: The `Gender` feature's category map.
        extracted_history_len_steps: Width of the history region, for datasets written before
            the layout was recorded in metadata.

    Returns:
        Tuple of (extracted, has_age, has_sex), each a set of patient-episode IDs.

    Raises:
        FileNotFoundError: If the fold has no extracted partitions.
    """
    from TransEHR2.data.preprocessing import load_dataset, load_episode_ids

    extracted, has_age, has_sex = set(), set(), set()
    for split in ('train', 'val', 'test'):
        base = os.path.join(data_dir, fold, split)
        if not os.path.exists(os.path.join(base, 'metadata.pkl')):
            continue
        dataset = load_dataset(base, extracted_history_len_steps=extracted_history_len_steps)
        ids = load_episode_ids(base, n_episodes=dataset.n_extracted_episodes)
        static = np.asarray(dataset.static_data)

        extracted.update(ids.tolist())
        has_age.update(ids[age_observed(static[:, offsets['Age']])].tolist())
        sex = decode_categorical(static[:, offsets['Gender']], category_map)
        has_sex.update(ids[sex != MISSING_LABEL].tolist())

    if not extracted:
        raise FileNotFoundError(
            f'no extracted partitions under {os.path.join(data_dir, fold)}, so feature '
            f'availability cannot be determined.'
        )
    return extracted, has_age, has_sex


def build_cohort(scored, extracted, has_age, has_sex):
    """The episodes the analysis runs on, and the funnel that produced them.

    An episode belongs to the cohort exactly when all three features exist: a Charlson index,
    an age and a sex. Defining it this way rather than by a predicate over the arrays is what
    lets both arms be handed one list -- see `TransEHR2.data.cohorts.manifest_mask`.

    Args:
        scored: IDs with a Charlson index.
        extracted: IDs present in the extracted arrays.
        has_age: IDs with an age on record.
        has_sex: IDs with a sex on record.

    Returns:
        Tuple of (cohort, funnel) where `cohort` is the sorted ID list and `funnel` is an
        ordered list of (label, count) explaining what each condition removed.
    """
    in_arrays = scored & extracted
    with_age = in_arrays & has_age
    with_both = with_age & has_sex
    funnel = [
        ('with a Charlson index', len(scored)),
        ('  and in the extracted arrays', len(in_arrays)),
        ('  and with an age on record', len(with_age)),
        ('  and with a sex on record', len(with_both)),
    ]
    return sorted(with_both), funnel


def write_cohort(path, cohort, description):
    """Write the episode manifest both arms are given.

    Args:
        path: Output path.
        cohort: Sorted patient-episode IDs.
        description: One line recorded as a comment, so a manifest found later says what it is.

    Returns:
        The path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(f'# {description}\n')
        handle.write('# One patient-episode id per line: patient_id * 1000 + episode_number.\n')
        handle.write(f'# {len(cohort)} episodes.\n')
        for episode_id in cohort:
            handle.write(f'{episode_id}\n')
    return path


def check_folds_agree(data_dir, fold_names, cohort, extracted_history_len_steps=None):
    """Confirm every fold's partitions cover the same cohort episodes.

    Folds are partitions of one episode set, so a manifest built from one fold has to be
    selectable in all of them. A fold that is missing some would train the control on fewer
    episodes than the regression, which is the failure this rules out.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold_names: Folds to check.
        cohort: The manifest's IDs.
        extracted_history_len_steps: As for `feature_availability`.

    Returns:
        Dict mapping fold name to the sorted IDs it does not carry.
    """
    from TransEHR2.data.preprocessing import load_dataset, load_episode_ids

    wanted = set(int(episode_id) for episode_id in cohort)
    missing = {}
    for fold in fold_names:
        covered = set()
        for split in ('train', 'val', 'test'):
            base = os.path.join(data_dir, fold, split)
            if not os.path.exists(os.path.join(base, 'metadata.pkl')):
                continue
            dataset = load_dataset(base,
                                   extracted_history_len_steps=extracted_history_len_steps)
            covered.update(load_episode_ids(
                base, n_episodes=dataset.n_extracted_episodes).tolist())
        if covered:
            missing[fold] = sorted(wanted - covered)
    return missing


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
    parser.add_argument('--write_cohort', nargs='?', const=DEFAULT_COHORT, default=None,
                        metavar='PATH',
                        help=f'Also write the episode manifest the analysis runs on -- the '
                             f'episodes with an index, an age and a sex. This is the file both '
                             f'arms are given (default path: {DEFAULT_COHORT})')
    parser.add_argument('--cohort_fold', default=None, metavar='FOLD',
                        help='Fold whose extracted arrays supply age and sex when building the '
                             'manifest (default: the first fold that has any)')
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

    if not args.write_cohort:
        return 0

    with open(dataset_config['VARIABLE_PROPERTIES_PATH']) as handle:
        variable_properties = yaml.safe_load(handle)
    static_feats = dataset_config['STATIC_FEATS']
    for required in ('Age', 'Gender'):
        if required not in static_feats:
            raise SystemExit(
                f'{args.dataset_config} does not list {required!r} in STATIC_FEATS, so the '
                f'extracted arrays do not carry it and the cohort cannot be defined on it.'
            )
    offsets = static_offsets(variable_properties, static_feats,
                             dataset_config.get('MAX_TOKEN_LENGTH', 0))
    extracted_history = dataset_config.get('MAX_HISTORY_LEN_STEPS')

    cohort_fold = args.cohort_fold or fold_names[0]
    print(f'\nBuilding the cohort manifest from {cohort_fold}...')
    extracted, has_age, has_sex = feature_availability(
        data_dir, cohort_fold, offsets, variable_properties['Gender'].get('category_map', {}),
        extracted_history,
    )
    cohort, funnel = build_cohort(set(frame['episode_id'].tolist()), extracted, has_age,
                                 has_sex)
    width = max(len(label) for label, _ in funnel)
    for label, count in funnel:
        print(f'  {label:{width}}  {count}')
    if not cohort:
        print('\nThe cohort is empty, so there is nothing to run on.', file=sys.stderr)
        return 1

    # Every fold has to be able to select the whole manifest, or the control would train on
    # fewer episodes in some fold than the regression scores.
    missing_by_fold = check_folds_agree(data_dir, fold_names, cohort, extracted_history)
    short = {fold: ids for fold, ids in missing_by_fold.items() if ids}
    for fold, ids in sorted(short.items()):
        shown = ', '.join(str(i) for i in ids[:20])
        more = '' if len(ids) <= 20 else f', ... ({len(ids)} total)'
        print(f'  {fold} does not carry {len(ids)} of them: {shown}{more}', file=sys.stderr)
    if short:
        print(
            '\nFolds are partitions of one episode set, so every fold must carry every cohort '
            'episode. Rebuild the manifest against a fold that does, or re-extract the folds '
            'that are short.',
            file=sys.stderr,
        )
        return 1
    print(f'  every fold carries all {len(cohort)} of them')

    path = write_cohort(
        args.write_cohort, cohort,
        'Charlson analysis cohort: episodes with a Charlson index, an age and a sex.',
    )
    print(f'\nWrote {len(cohort)} episode ids to {path}')
    print('Name this file in COHORT_EPISODES for the control experiment, and pass it to '
          'run_charlson_logistic_regression.py --cohort_episodes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
