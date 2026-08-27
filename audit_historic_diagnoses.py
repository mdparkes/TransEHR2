#!/usr/bin/env python
"""
Audit how often a last ICU stay's phenotype labels are already present in the
patient's historical diagnosis text.

Each patient-episode's timeseries CSV carries a "Diagnosis Descriptions" text
feature. That feature is a pipe-delimited string of ICD long titles, emitted
once per hospital admission and timestamped at that admission's discharge time.
Descriptions belonging to the admission that contains the current ICU stay are
blanked out upstream, so the remaining descriptions at negative timestamps
(Hours < 0) are the diagnoses of the patient's *previous* hospitalizations --
the "historical diagnosis text" that TransEHR2 consumes as an input feature.

Phenotype labels, in contrast, are derived from the ICD codes assigned during
the current ICU stay (see TransEHR2/data/create_phenotypes.py). Because the
experiments retain only the last ICU stay per patient, many of those patients
have prior admissions whose diagnosis text may already name the very phenotype
being predicted. This script quantifies that overlap.

Method
------
1. Collect the unique patient-episodes named in the fold listfiles (episodes are
   deduplicated across folds and partitions, so each stay is audited once).
2. Read the positive phenotype labels for each episode from the
   `phenotyping_{partition}_listfile.csv` files.
3. Read each episode's timeseries CSV, keep rows with Hours < 0, and split the
   "Diagnosis Descriptions" strings on '|' into individual ICD long titles.
4. Map each long title back to an ICD code via the patient's own
   `diagnoses.csv` (which carries LONG_TITLE alongside ICD_CODE and
   ICD_VERSION), then map that code to its HCUP phenotype group(s) using the
   same definition YAMLs that produced the labels.
5. Intersect each episode's positive labels with the phenotype groups recovered
   from its historical diagnosis text.

Note that the audit is strict about category names by default: ICD-9 codes
reach their phenotype through HCUP CCS 2015 and ICD-10 codes through HCUP CCSR
2024, and the two vocabularies name one benchmark category differently
("Congestive heart failure; nonhypertensive" vs. "Heart failure"). Pass
--merge_synonymous_phenotypes to count those as the same phenotype.

Outputs (written to --output_dir)
---------------------------------
* `historic_diagnosis_audit_summary.csv`   -- overall label- and episode-level counts
* `historic_diagnosis_audit_by_phenotype.csv` -- one row per benchmark phenotype
* `historic_diagnosis_audit_per_episode.csv`  -- one row per stay (with --write_per_episode)

Usage:
    python audit_historic_diagnoses.py TransEHR2/configs/datasets/mimic4.yaml -w 8
    python audit_historic_diagnoses.py TransEHR2/configs/datasets/mimic4.yaml --unfiltered
    python audit_historic_diagnoses.py TransEHR2/configs/datasets/mimic4.yaml --folds fold0
"""

import argparse
import os
import re
import sys
import yaml

import pandas as pd

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

TEXT_FEATURE = 'Diagnosis Descriptions'
TASK_PREFIX = 'phenotyping'

# ICD-9 codes reach their phenotype through HCUP CCS 2015 and ICD-10 codes
# through HCUP CCSR 2024. The two vocabularies name one benchmark category
# differently, so a heart failure diagnosis coded in ICD-9 will not match a
# heart failure label derived from an ICD-10 code unless the two names are
# treated as equivalent. Enable that with --merge_synonymous_phenotypes.
PHENOTYPE_SYNONYMS = [
    {'Congestive heart failure; nonhypertensive', 'Heart failure'},
]


# ---------------------------------------------------------------------------
# Phenotype definitions
# ---------------------------------------------------------------------------

def load_code_to_groups(icd9_definitions_path, icd10_definitions_path):
    """Build a {(icd_version, icd_code): [phenotype group, ...]} lookup.

    Mirrors the mapping built by TransEHR2/data/create_phenotypes.py so that
    groups recovered from historical diagnoses are named exactly as the label
    columns of the phenotyping listfiles.
    """
    code_to_groups = {}
    for version, path in ((9, icd9_definitions_path), (10, icd10_definitions_path)):
        with open(path, 'r') as f:
            definitions = yaml.safe_load(f)
        for group, spec in definitions.items():
            for code in spec['codes']:
                code_to_groups.setdefault((version, str(code)), []).append(group)
    return code_to_groups


# ---------------------------------------------------------------------------
# Listfile collection
# ---------------------------------------------------------------------------

def discover_folds(data_dir, requested_folds=None):
    """Discover fold directories, matching extract_data.py's pattern."""
    if requested_folds:
        return requested_folds
    return sorted([
        item for item in os.listdir(data_dir)
        if re.match(r'fold\d+', item)
        and os.path.isdir(os.path.join(data_dir, item))
    ])


def listfile_path(fold_dir, name, unfiltered):
    """Return the path to a listfile, preferring the .unfiltered.csv backup.

    filter_listfiles_by_discharge_summary.py renames the full-dataset listfiles
    to {name}.unfiltered.csv before writing the discharge-summary subset. Pass
    unfiltered=True to audit the full dataset even when the subset is in place.
    """
    path = os.path.join(fold_dir, f'{name}.csv')
    if unfiltered:
        backup = os.path.join(fold_dir, f'{name}.unfiltered.csv')
        if os.path.exists(backup):
            return backup
    return path


def episode_key(path):
    """Normalise a stay path to the episode CSV path used as the join key."""
    return os.path.abspath(re.sub(r'_timeseries\.csv$', '.csv', str(path)))


def collect_episodes_and_labels(data_dir, fold_names, unfiltered):
    """Collect the unique episodes in the fold listfiles and their labels.

    Returns:
        Tuple of (episodes, labels, phenotype_names, n_listfile_rows) where
        `episodes` maps an episode key to (patient_id, episode_number),
        `labels` maps an episode key to the set of positive phenotype names,
        `phenotype_names` is the ordered list of label columns, and
        `n_listfile_rows` is the total number of rows read (before dedup).
    """
    episodes = {}
    labels = {}
    phenotype_names = None
    n_listfile_rows = 0

    for fold_name in fold_names:
        fold_dir = os.path.join(data_dir, fold_name)
        for partition in ('train', 'val', 'test'):
            dataset_file = listfile_path(fold_dir, f'{fold_name}_{partition}', unfiltered)
            pheno_file = listfile_path(fold_dir, f'{TASK_PREFIX}_{partition}_listfile', unfiltered)
            if not os.path.exists(dataset_file):
                if partition != 'val':
                    print(f"  WARNING: missing {dataset_file}", file=sys.stderr)
                continue
            if not os.path.exists(pheno_file):
                print(f"  WARNING: missing {pheno_file}", file=sys.stderr)
                continue

            dataset_df = pd.read_csv(dataset_file)
            n_listfile_rows += len(dataset_df)
            for path, pt_id, ep_num in zip(dataset_df.iloc[:, 0],
                                           dataset_df.iloc[:, 1].astype(int),
                                           dataset_df.iloc[:, 2].astype(int)):
                episodes[episode_key(path)] = (int(pt_id), int(ep_num))

            pheno_df = pd.read_csv(pheno_file, index_col=0)
            pheno_df = pheno_df.drop(columns=['period_length'], errors='ignore')
            if phenotype_names is None:
                phenotype_names = list(pheno_df.columns)
            elif list(pheno_df.columns) != phenotype_names:
                raise ValueError(
                    f"Phenotype columns in {pheno_file} differ from earlier listfiles."
                )
            for stay_path, row in zip(pheno_df.index, pheno_df.to_numpy()):
                labels[episode_key(stay_path)] = {
                    name for name, value in zip(phenotype_names, row) if int(value) == 1
                }

    if phenotype_names is None:
        raise FileNotFoundError("No phenotyping listfiles were found.")

    return episodes, labels, phenotype_names, n_listfile_rows


# ---------------------------------------------------------------------------
# Per-patient audit worker
# ---------------------------------------------------------------------------

def build_title_to_groups(diagnoses_df, code_to_groups):
    """Map each ICD long title in a patient's diagnoses to phenotype groups.

    Groups are resolved from (ICD_VERSION, ICD_CODE) via the HCUP definition
    YAMLs rather than from the HCUP_CCS_2015 / HCUP_CCSR_2024 columns of
    diagnoses.csv. Those columns join multiple categories with '; ', which is
    itself part of several CCS category names (e.g. "Congestive heart failure;
    nonhypertensive"), so they cannot be parsed unambiguously.
    """
    title_to_groups = defaultdict(set)

    for row in diagnoses_df.itertuples(index=False):
        title = getattr(row, 'LONG_TITLE', None)
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            version = int(getattr(row, 'ICD_VERSION'))
        except (AttributeError, TypeError, ValueError):
            continue
        code = str(getattr(row, 'ICD_CODE', '')).strip()
        title_to_groups[title.strip()] |= set(code_to_groups.get((version, code), []))

    return title_to_groups


def historical_titles(episode_csv_path):
    """Return the ICD long titles in an episode's pre-admission diagnosis text.

    Returns:
        Tuple of (titles, n_text_records, status) where `titles` is the list of
        long titles parsed out of the historical (Hours < 0) diagnosis strings,
        `n_text_records` is the number of historical records that carried text
        (one per previous hospital admission), and `status` is 'ok',
        'missing_timeseries', or 'missing_column'.
    """
    ts_path = re.sub(r'\.csv$', '_timeseries.csv', episode_csv_path)
    try:
        df = pd.read_csv(ts_path, usecols=['Hours', TEXT_FEATURE])
    except FileNotFoundError:
        return [], 0, 'missing_timeseries'
    except ValueError:
        # usecols raises ValueError when a requested column is absent
        return [], 0, 'missing_column'

    historical = df.loc[df['Hours'] < 0, TEXT_FEATURE].dropna()
    historical = historical.astype(str).str.strip()
    historical = historical[historical != '']

    titles = []
    for text in historical:
        titles.extend(t.strip() for t in text.split('|') if t.strip())
    return titles, int(len(historical)), 'ok'


# The ICD code -> phenotype group lookup holds ~90k entries. It is shared with
# the workers once at pool start-up rather than pickled with every task.
_CODE_TO_GROUPS = {}


def _init_worker(code_to_groups):
    global _CODE_TO_GROUPS
    _CODE_TO_GROUPS = code_to_groups


def audit_patient(task):
    """Audit every selected episode of one patient.

    Args:
        task: Tuple of (patient_dir, episode_records) where episode_records is
            a list of (episode_key, episode_number).

    Returns:
        List of per-episode result dicts.
    """
    patient_dir, episode_records = task
    code_to_groups = _CODE_TO_GROUPS

    diagnoses_path = os.path.join(patient_dir, 'diagnoses.csv')
    if os.path.exists(diagnoses_path):
        diagnoses_df = pd.read_csv(
            diagnoses_path, dtype={'ICD_CODE': str, 'ICD_VERSION': 'Int64'}
        )
        title_to_groups = build_title_to_groups(diagnoses_df, code_to_groups)
    else:
        title_to_groups = {}

    # Number of ICU stays on record for this patient. Episodes are numbered
    # 1..n_stays in INTIME order (see MIMICDataReader.get_stays_data), so the
    # last stay is the one whose episode number equals n_stays.
    stays_path = os.path.join(patient_dir, 'stays.csv')
    try:
        n_stays = len(pd.read_csv(stays_path, usecols=['INTIME']))
    except (FileNotFoundError, ValueError):
        n_stays = None

    results = []
    for key, episode_number in episode_records:
        titles, n_text_records, status = historical_titles(key)
        unique_titles = set(titles)
        groups = set()
        n_unmapped_titles = 0
        for title in unique_titles:
            mapped = title_to_groups.get(title)
            if mapped:
                groups |= mapped
            elif mapped is None:
                n_unmapped_titles += 1
        results.append({
            'episode_key': key,
            'episode_number': episode_number,
            'n_stays': n_stays,
            'is_last_stay': None if n_stays is None else bool(episode_number == n_stays),
            'status': status,
            'n_historical_dx_records': n_text_records,
            'n_historical_dx_titles': len(titles),
            'n_unique_historical_dx_titles': len(unique_titles),
            'n_unmapped_historical_dx_titles': n_unmapped_titles,
            'historical_groups': groups,
        })
    return results


def run_audit(episodes, code_to_groups, n_workers):
    """Audit all episodes, grouping the work by patient."""
    by_patient = defaultdict(list)
    for key, (pt_id, ep_num) in episodes.items():
        by_patient[os.path.dirname(key)].append((key, ep_num))

    tasks = [
        (patient_dir, sorted(records))
        for patient_dir, records in sorted(by_patient.items())
    ]

    results = []
    if n_workers <= 1:
        _init_worker(code_to_groups)
        for i, task in enumerate(tasks, 1):
            results.extend(audit_patient(task))
            if i % 500 == 0:
                print(f"    {i}/{len(tasks)} patients", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(code_to_groups,)
        ) as executor:
            futures = [executor.submit(audit_patient, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                results.extend(future.result())
                if i % 500 == 0:
                    print(f"    {i}/{len(tasks)} patients", flush=True)
    return results


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

def expand_synonyms(groups):
    """Add the equivalent names of any phenotype in `groups` (see PHENOTYPE_SYNONYMS)."""
    expanded = set(groups)
    for synonyms in PHENOTYPE_SYNONYMS:
        if expanded & synonyms:
            expanded |= synonyms
    return expanded


def pct(numerator, denominator):
    return float('nan') if denominator == 0 else 100.0 * numerator / denominator


def tabulate(results, labels, phenotype_names, merge_synonyms=False):
    """Cross the historical diagnosis groups with the phenotype labels.

    Returns:
        Tuple of (per_episode_df, by_phenotype_df, summary_df).
    """
    per_episode_rows = []
    # Per-phenotype tallies keyed by phenotype name.
    pos = defaultdict(int)            # episodes where the label is positive
    pos_matched = defaultdict(int)    # ...and the phenotype is in the history text
    pos_with_hist = defaultdict(int)  # positive episodes that have any history text
    pos_matched_with_hist = defaultdict(int)
    neg = defaultdict(int)            # episodes where the label is negative
    neg_matched = defaultdict(int)    # ...but the phenotype is in the history text

    for row in results:
        key = row['episode_key']
        positives = labels.get(key)
        if positives is None:
            row = dict(row, status='missing_labels')
            per_episode_rows.append(row)
            continue

        history = row['historical_groups']
        if merge_synonyms:
            history = expand_synonyms(history)
        has_history = row['n_historical_dx_records'] > 0
        matched = positives & history

        for name in phenotype_names:
            if name in positives:
                pos[name] += 1
                if has_history:
                    pos_with_hist[name] += 1
                if name in history:
                    pos_matched[name] += 1
                    if has_history:
                        pos_matched_with_hist[name] += 1
            else:
                neg[name] += 1
                if name in history:
                    neg_matched[name] += 1

        per_episode_rows.append({
            **row,
            'n_positive_labels': len(positives),
            'n_positive_labels_in_history': len(matched),
            'has_historical_dx_text': has_history,
            'any_label_in_history': len(matched) > 0,
            'all_labels_in_history': len(positives) > 0 and len(matched) == len(positives),
            'positive_labels': '|'.join(sorted(positives)),
            'positive_labels_in_history': '|'.join(sorted(matched)),
        })

    per_episode_df = pd.DataFrame(per_episode_rows)
    if 'historical_groups' in per_episode_df.columns:
        per_episode_df = per_episode_df.drop(columns=['historical_groups'])

    by_phenotype_df = pd.DataFrame([{
        'phenotype': name,
        'n_stays_positive': pos[name],
        'n_positive_in_history': pos_matched[name],
        'pct_positive_in_history': pct(pos_matched[name], pos[name]),
        'n_stays_positive_with_history': pos_with_hist[name],
        'n_positive_in_history_given_history': pos_matched_with_hist[name],
        'pct_positive_in_history_given_history': pct(pos_matched_with_hist[name], pos_with_hist[name]),
        'n_stays_negative': neg[name],
        'n_negative_in_history': neg_matched[name],
        'pct_negative_in_history': pct(neg_matched[name], neg[name]),
    } for name in phenotype_names])
    by_phenotype_df = by_phenotype_df.sort_values(
        ['pct_positive_in_history', 'n_stays_positive'], ascending=False
    ).reset_index(drop=True)

    audited = per_episode_df[per_episode_df['status'] == 'ok'] if len(per_episode_df) else per_episode_df
    n_labels = int(audited['n_positive_labels'].sum()) if len(audited) else 0
    n_labels_matched = int(audited['n_positive_labels_in_history'].sum()) if len(audited) else 0
    with_hist = audited[audited['has_historical_dx_text']] if len(audited) else audited
    n_labels_hist = int(with_hist['n_positive_labels'].sum()) if len(with_hist) else 0
    n_labels_hist_matched = int(with_hist['n_positive_labels_in_history'].sum()) if len(with_hist) else 0
    labelled = audited[audited['n_positive_labels'] > 0] if len(audited) else audited

    summary_rows = [
        ('stays_audited', len(audited)),
        ('stays_with_historical_dx_text', int(audited['has_historical_dx_text'].sum()) if len(audited) else 0),
        ('pct_stays_with_historical_dx_text',
         pct(int(audited['has_historical_dx_text'].sum()) if len(audited) else 0, len(audited))),
        ('stays_with_at_least_one_positive_label', len(labelled)),
        ('positive_labels_total', n_labels),
        ('positive_labels_in_history', n_labels_matched),
        ('pct_positive_labels_in_history', pct(n_labels_matched, n_labels)),
        ('positive_labels_total_stays_with_history', n_labels_hist),
        ('positive_labels_in_history_stays_with_history', n_labels_hist_matched),
        ('pct_positive_labels_in_history_stays_with_history',
         pct(n_labels_hist_matched, n_labels_hist)),
        ('stays_with_any_label_in_history',
         int(labelled['any_label_in_history'].sum()) if len(labelled) else 0),
        ('pct_stays_with_any_label_in_history',
         pct(int(labelled['any_label_in_history'].sum()) if len(labelled) else 0, len(labelled))),
        ('stays_with_all_labels_in_history',
         int(labelled['all_labels_in_history'].sum()) if len(labelled) else 0),
        ('pct_stays_with_all_labels_in_history',
         pct(int(labelled['all_labels_in_history'].sum()) if len(labelled) else 0, len(labelled))),
        ('mean_positive_labels_per_stay',
         float(audited['n_positive_labels'].mean()) if len(audited) else float('nan')),
        ('unmapped_historical_dx_titles',
         int(audited['n_unmapped_historical_dx_titles'].sum()) if len(audited) else 0),
    ]
    if len(per_episode_df):
        for status, count in per_episode_df['status'].value_counts().items():
            if status != 'ok':
                summary_rows.append((f'stays_skipped_{status}', int(count)))
        if per_episode_df['is_last_stay'].notna().any():
            checked = per_episode_df['is_last_stay'].dropna()
            summary_rows.append(('stays_verified_as_patient_last_stay', int(checked.sum())))
            summary_rows.append(('stays_not_patient_last_stay', int((~checked.astype(bool)).sum())))

    summary_df = pd.DataFrame(summary_rows, columns=['metric', 'value'])
    # Keep counts as ints and percentages as floats rather than letting pandas
    # coerce the whole column to float64.
    summary_df['value'] = pd.Series([v for _, v in summary_rows], dtype=object)
    return per_episode_df, by_phenotype_df, summary_df


def print_report(summary_df, by_phenotype_df, merge_synonyms=False):
    print(f"\n{'='*78}")
    print("Historical diagnosis text vs. last-stay phenotype labels")
    print(f"{'='*78}\n")
    if merge_synonyms:
        print("  (ICD-9/ICD-10 synonymous phenotype names merged for matching)\n")
    for metric, value in summary_df.itertuples(index=False):
        if isinstance(value, float):
            print(f"  {metric:<56s} {value:>12.2f}")
        else:
            print(f"  {metric:<56s} {value:>12d}")


    print(f"\n{'-'*78}")
    print("By phenotype (sorted by % of positive stays whose label is in the history text)")
    print(f"{'-'*78}")
    header = f"  {'phenotype':<52s} {'pos':>7s} {'in hist':>8s} {'%':>7s}"
    print(header)
    for row in by_phenotype_df.itertuples(index=False):
        print(f"  {row.phenotype[:52]:<52s} {row.n_stays_positive:>7d} "
              f"{row.n_positive_in_history:>8d} {row.pct_positive_in_history:>7.1f}")
    print()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=("Audit how many of a last ICU stay's phenotype labels already "
                     "appear in the patient's historical diagnosis text.")
    )
    parser.add_argument(
        'dataset_config', type=str,
        help="YAML file specifying dataset parameters (e.g. TransEHR2/configs/datasets/mimic4.yaml)"
    )
    parser.add_argument(
        '--folds', type=str, nargs='*', default=None,
        help="Specific folds to read listfiles from (default: all folds). Episodes "
             "are deduplicated across folds and partitions."
    )
    parser.add_argument(
        '--unfiltered', action='store_true',
        help="Read the .unfiltered.csv listfile backups when present, i.e. audit the "
             "full dataset rather than the discharge-summary subset."
    )
    parser.add_argument(
        '--n_workers', '-w', type=int, default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help="Directory for the output CSV files (default: current directory)"
    )
    parser.add_argument(
        '--merge_synonymous_phenotypes', action='store_true',
        help="Treat benchmark categories that HCUP CCS 2015 and CCSR 2024 name "
             "differently as equivalent when matching (see PHENOTYPE_SYNONYMS). "
             "Off by default, which reports the strict name-for-name overlap."
    )
    parser.add_argument(
        '--write_per_episode', action='store_true',
        help="Also write the per-stay audit table."
    )
    parser.add_argument(
        '--icd9_phenotype_definitions', '-p9', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'hcup_ccs_2015_definitions.yaml'),
        help="YAML file with ICD-9 phenotype definitions."
    )
    parser.add_argument(
        '--icd10_phenotype_definitions', '-p10', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'hcup_ccsr_2024_definitions.yaml'),
        help="YAML file with ICD-10 phenotype definitions."
    )
    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    data_dir = config['DATA_DIR']

    fold_names = discover_folds(data_dir, args.folds)
    if not fold_names:
        print("No fold directories found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(fold_names)} fold(s): {fold_names}")
    if args.unfiltered:
        print("Reading .unfiltered.csv listfiles where available.")

    code_to_groups = load_code_to_groups(
        args.icd9_phenotype_definitions, args.icd10_phenotype_definitions
    )

    print("\nCollecting episodes and phenotype labels from listfiles...")
    episodes, labels, phenotype_names, n_rows = collect_episodes_and_labels(
        data_dir, fold_names, args.unfiltered
    )
    print(f"  {n_rows} listfile rows -> {len(episodes)} unique stays, "
          f"{len(phenotype_names)} phenotypes")

    print(f"\nAuditing historical diagnosis text with {args.n_workers} worker(s)...")
    results = run_audit(episodes, code_to_groups, args.n_workers)

    per_episode_df, by_phenotype_df, summary_df = tabulate(
        results, labels, phenotype_names, args.merge_synonymous_phenotypes
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'historic_diagnosis_audit_summary.csv')
    phenotype_path = os.path.join(args.output_dir, 'historic_diagnosis_audit_by_phenotype.csv')
    summary_df.to_csv(summary_path, index=False)
    by_phenotype_df.to_csv(phenotype_path, index=False)
    written = [summary_path, phenotype_path]
    if args.write_per_episode:
        episode_path = os.path.join(args.output_dir, 'historic_diagnosis_audit_per_episode.csv')
        per_episode_df.to_csv(episode_path, index=False)
        written.append(episode_path)

    print_report(summary_df, by_phenotype_df, args.merge_synonymous_phenotypes)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == '__main__':
    main()
