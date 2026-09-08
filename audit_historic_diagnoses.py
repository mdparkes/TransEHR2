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

Pass --cohort (or --cohort-episodes) to restrict the audit to the episodes an
experiment actually ran on. Without it the audit describes the listfile cohort,
which is a superset: the extraction drops episodes that fail the minimum-length
criterion, and the revision experiments restrict themselves further to a named
cohort. A supplementary table should describe the population that was evaluated.

Note that a named cohort is an array-side predicate: 'diagnosis_history' selects
episodes whose *extracted* arrays carry a pre-admission diagnosis-descriptions
record, whereas this audit reads the source CSVs. The two can disagree, and
TransEHR2/data/cohorts.py documents why. Restricting to a cohort makes the
audited population match the experiment; it does not make the two definitions
identical.

Usage:
    python audit_historic_diagnoses.py TransEHR2/configs/datasets/mimic4.yaml -w 8
    python audit_historic_diagnoses.py TransEHR2/configs/datasets/mimic4.yaml --unfiltered
    python audit_historic_diagnoses.py TransEHR2/configs/datasets/mimic4.yaml --folds fold0
"""

import argparse
import os
import pickle
import re
import sys
import yaml

import numpy as np
import pandas as pd

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from TransEHR2.data.cohorts import COHORTS, cohort_indices
from TransEHR2.data.preprocessing import load_episode_ids

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


def cohort_episode_id_set(data_dir, fold_names, cohort, manifest):
    """Patient-episode IDs a cohort keeps, unioned over every fold and partition.

    Cohort membership is a property of an episode's extracted arrays, so an
    episode's status is the same wherever it lands in the cross-validation
    splits; taking the union simply collects every episode that was extracted
    somewhere. Delegates the predicate itself to `TransEHR2.data.cohorts`, so
    this selects exactly what an experiment configured the same way selects.

    Args:
        data_dir: Root directory holding the fold subdirectories.
        fold_names: Fold directory names to scan.
        cohort: A name in `COHORTS`, or None.
        manifest: An explicit episode manifest, or None.

    Returns:
        Set of patient-episode IDs, or None when no restriction is requested.
    """
    if cohort is None and manifest is None:
        return None

    selected = set()
    partitions_read = 0
    for fold_name in fold_names:
        for partition in ('train', 'val', 'test'):
            part_dir = os.path.join(data_dir, fold_name, partition)
            if not os.path.isdir(part_dir):
                continue
            try:
                episode_ids = load_episode_ids(part_dir)
            except FileNotFoundError as exc:
                print(f'  WARNING: {exc}', file=sys.stderr)
                continue

            def load_mmap(name):
                return np.load(os.path.join(part_dir, f'{name}.npy'), mmap_mode='r')

            with open(os.path.join(part_dir, 'metadata.pkl'), 'rb') as handle:
                metadata = pickle.load(handle)
            max_history = metadata.get('max_history_len_steps')
            if max_history is None:
                raise ValueError(
                    f'{part_dir}/metadata.pkl does not record max_history_len_steps, so '
                    'cohort membership cannot be computed against this extraction.'
                )

            indices = cohort_indices(
                {
                    'val_masks': load_mmap('val_masks'),
                    'val_text_indicators': load_mmap('val_text_indicators'),
                    'max_history_len_steps': int(max_history),
                },
                cohort,
                episode_ids=episode_ids,
                manifest=manifest,
            )
            kept = episode_ids if indices is None else episode_ids[indices]
            selected.update(int(i) for i in kept)
            partitions_read += 1

    if not partitions_read:
        raise FileNotFoundError(
            f'no extracted partitions found under {data_dir} for folds {fold_names}, so '
            'the cohort cannot be resolved. Extract the data first, or drop --cohort.'
        )
    return selected


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


def resolve_history_columns(episode_csv_path, feature_names):
    """Columns that decide which timesteps enter the extraction's merged frame.

    `filter_timeseries_records` truncates the outer merge of the
    value-associated and text features, having first dropped rows that are
    empty across each. Timesteps carrying only event-associated features never
    enter that ordering, so counting every row of the episode CSV would place
    a diagnosis record further back than extraction does. Vector-valued
    features occupy one column per dimension, named `feature_0`, `feature_1`
    and so on, matching `MIMICDataReader._get_feature_column_names`.

    Args:
        episode_csv_path: Any episode CSV; the schema is shared across episodes.
        feature_names: Base names of the value-associated and text features.

    Returns:
        List of column names, empty if the header cannot be read.
    """
    ts_path = re.sub(r'\.csv$', '_timeseries.csv', str(episode_csv_path))
    try:
        header = pd.read_csv(ts_path, nrows=0)
    except (FileNotFoundError, ValueError):
        return []
    columns = []
    for base in feature_names:
        pattern = re.compile(f'^{re.escape(base)}(_\\d+)?$')
        columns.extend(column for column in header.columns
                       if pattern.match(column))
    return columns


def historical_titles(episode_csv_path):
    """Return the ICD long titles in an episode's pre-admission diagnosis text.

    Extraction keeps only the most recent `max_history_len_steps` pre-admission
    timesteps, so a diagnosis record further back than that exists in the
    source CSV but never reaches the model. Both figures are returned: the
    truncated one is what the model could have read, and the untruncated one is
    what the source data holds, so the difference between them can be reported
    rather than assumed negligible.

    Returns:
        Tuple of (titles, n_text_records, titles_all, n_text_records_all,
        status), where the first pair is confined to the retained window and
        the second covers every pre-admission record. `status` is 'ok',
        'missing_timeseries' or 'missing_column'.
    """
    limit = _HISTORY_LIMIT
    columns = _HISTORY_COLUMNS

    ts_path = re.sub(r'\.csv$', '_timeseries.csv', str(episode_csv_path))
    wanted = {'Hours', TEXT_FEATURE, *columns}
    try:
        df = pd.read_csv(ts_path, usecols=lambda column: column in wanted)
    except FileNotFoundError:
        return [], 0, [], 0, 'missing_timeseries'
    if 'Hours' not in df.columns or TEXT_FEATURE not in df.columns:
        return [], 0, [], 0, 'missing_column'

    history = df.loc[df['Hours'] < 0].sort_values('Hours')

    # Restrict to the timesteps extraction would have ordered, so that the
    # window counts the same records it does.
    present = [column for column in columns if column in history.columns]
    if present:
        history = history.dropna(how='all', subset=present)

    def titles_of(frame):
        text = frame[TEXT_FEATURE].dropna().astype(str).str.strip()
        text = text[text != '']
        titles = []
        for value in text:
            titles.extend(t.strip() for t in value.split('|') if t.strip())
        return titles, int(len(text))

    titles_all, n_all = titles_of(history)
    if limit is not None and len(history) > limit:
        titles, n_kept = titles_of(history.iloc[-limit:])
    else:
        titles, n_kept = titles_all, n_all

    return titles, n_kept, titles_all, n_all, 'ok'


# The ICD code -> phenotype group lookup holds ~90k entries. It is shared with
# the workers once at pool start-up rather than pickled with every task.
_CODE_TO_GROUPS = {}
# Extraction's pre-admission window, and the columns that decide which
# timesteps it orders. None disables truncation.
_HISTORY_LIMIT = None
_HISTORY_COLUMNS = ()


def _init_worker(code_to_groups, history_limit=None, history_columns=()):
    global _CODE_TO_GROUPS, _HISTORY_LIMIT, _HISTORY_COLUMNS
    _CODE_TO_GROUPS = code_to_groups
    _HISTORY_LIMIT = history_limit
    _HISTORY_COLUMNS = tuple(history_columns)


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
        (titles, n_text_records, titles_all, n_text_records_all,
         status) = historical_titles(key)
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
            'n_historical_dx_records_untruncated': n_text_records_all,
            'n_historical_dx_titles': len(titles),
            'n_historical_dx_titles_untruncated': len(titles_all),
            'n_unique_historical_dx_titles': len(unique_titles),
            'n_unmapped_historical_dx_titles': n_unmapped_titles,
            'historical_groups': groups,
        })
    return results


def run_audit(episodes, code_to_groups, n_workers, history_limit=None,
              history_columns=()):
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
        _init_worker(code_to_groups, history_limit, history_columns)
        for i, task in enumerate(tasks, 1):
            results.extend(audit_patient(task))
            if i % 500 == 0:
                print(f"    {i}/{len(tasks)} patients", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
            initargs=(code_to_groups, history_limit, history_columns)
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

    if len(audited) and 'n_historical_dx_records_untruncated' in audited.columns:
        # What the pre-admission window removed. A stay that loses every
        # diagnosis record has text in the source data that the model never
        # saw, and would be counted as carrying history by an audit that
        # ignored the window.
        had_any = audited['n_historical_dx_records_untruncated'] > 0
        lost_all = had_any & (audited['n_historical_dx_records'] == 0)
        # Counts any loss, so it includes the stays that lost every title.
        lost_any = (audited['n_historical_dx_titles_untruncated']
                    > audited['n_historical_dx_titles'])
        dropped = int((audited['n_historical_dx_titles_untruncated']
                       - audited['n_historical_dx_titles']).sum())
        summary_rows += [
            ('stays_with_dx_text_before_truncation', int(had_any.sum())),
            ('stays_losing_all_dx_text_to_truncation', int(lost_all.sum())),
            ('stays_losing_any_dx_titles_to_truncation', int(lost_any.sum())),
            ('dx_titles_dropped_by_truncation', dropped),
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
        if isinstance(value, bool) or isinstance(value, str):
            print(f"  {metric:<56s} {value:>12s}")
        elif isinstance(value, float):
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
        '--no_truncate_history', action='store_true',
        help="Count every pre-admission diagnosis record, including those "
             "extraction discards. By default the audit keeps only the most "
             "recent MAX_HISTORY_LEN_STEPS pre-admission timesteps, matching "
             "what the model can actually read; this reports what the source "
             "data holds instead."
    )
    parser.add_argument(
        '--cohort', type=str, default=None, choices=list(COHORTS),
        help="Restrict the audit to the episodes a named cohort keeps, so that it "
             "describes the population an experiment configured the same way ran on. "
             "Requires the extracted arrays."
    )
    parser.add_argument(
        '--cohort_episodes', type=str, default=None,
        help="Restrict the audit to an explicit episode manifest (one patient-episode ID "
             "per line), as compute_charlson_index.py --write_cohort writes. Combines with "
             "--cohort: an episode must then satisfy both."
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
    history_limit = (None if args.no_truncate_history
                     else config.get('MAX_HISTORY_LEN_STEPS'))
    history_feature_names = (list(config.get('VALUED_FEATS', []))
                             + list(config.get('TEXT_FEATS', [])))

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

    n_listfile_stays = len(episodes)
    cohort_ids = cohort_episode_id_set(data_dir, fold_names, args.cohort,
                                       args.cohort_episodes)
    if cohort_ids is not None:
        label = args.cohort or 'manifest'
        if args.cohort and args.cohort_episodes:
            label = f'{args.cohort} + manifest'
        episodes = {
            key: value for key, value in episodes.items()
            if value[0] * 1000 + value[1] in cohort_ids
        }
        labels = {key: value for key, value in labels.items() if key in episodes}
        print(f"  cohort '{label}' keeps {len(episodes)} of {n_listfile_stays} stays "
              f"({len(cohort_ids)} episodes in the cohort overall)")
        if not episodes:
            print('The cohort selects none of the listfile stays.', file=sys.stderr)
            sys.exit(1)

    # The episode CSVs share a schema, so the columns are resolved once.
    history_columns = ()
    if history_limit is not None and episodes:
        history_columns = tuple(resolve_history_columns(
            next(iter(episodes)), history_feature_names
        ))
        print(f"  pre-admission window: most recent {history_limit} timesteps "
              f"ordered over {len(history_columns)} value and text columns")
    else:
        print("  pre-admission window: none, counting every historical record")

    print(f"\nAuditing historical diagnosis text with {args.n_workers} worker(s)...")
    results = run_audit(episodes, code_to_groups, args.n_workers,
                        history_limit, history_columns)

    per_episode_df, by_phenotype_df, summary_df = tabulate(
        results, labels, phenotype_names, args.merge_synonymous_phenotypes
    )

    window = 'none' if history_limit is None else int(history_limit)
    restriction = args.cohort or ''
    if args.cohort_episodes:
        restriction = (f'{restriction} + manifest' if restriction else 'manifest')
    summary_df = pd.concat([
        pd.DataFrame([
            ('cohort_restriction', restriction or 'none'),
            ('history_window_steps', window),
            ('stays_in_listfiles', n_listfile_stays),
            ('stays_after_cohort_restriction', len(episodes)),
        ], columns=['metric', 'value']),
        summary_df,
    ], ignore_index=True)

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
