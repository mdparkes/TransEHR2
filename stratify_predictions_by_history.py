#!/usr/bin/env python3
"""Test whether diagnosis predictions benefit from carry-forward in the
historical diagnosis text.

``audit_historic_diagnoses.py`` establishes that for the final ICU stays of
patients with an earlier hospital admission, a majority of the phenotype
labels are already named in the diagnosis text of those earlier admissions.
This script asks the follow-up question: does the model actually predict
those labels better?

The comparison is made *within a single model and a single set of stays*, so
it is not confounded by the differences in sequence length, capacity and
optimization that separate the history and no-history experiments. Among
stays that have earlier diagnosis text, every true-positive label falls into
one of two groups:

    named    -- the earlier text already names this diagnosis
    unnamed  -- it does not

If the model exploits the carry-forward, the scores it assigns to `named`
positives should exceed those it assigns to `unnamed` positives. Stays with
no earlier text at all are reported as a third group for context, but they
are not part of the primary contrast because they differ in ways beyond the
presence of the label in the text.

Two effect measures are reported, both per diagnosis and pooled over all of
them, as a mean over folds with the standard error of that mean:

    P(named > unnamed)  -- the probability that a randomly chosen `named`
        positive receives a higher score than a randomly chosen `unnamed`
        positive, with ties counted as half. This is the Mann-Whitney
        statistic scaled to the unit interval. 0.5 means the model does not
        distinguish the two groups at all; 1.0 would mean perfect separation.
        It is threshold-free and invariant to any monotone recalibration.

    Recall difference -- the difference in recall between the two groups at
        a prevalence-matched threshold, which is easier to interpret but
        depends on the operating point.

Row alignment
-------------
The prediction CSVs written by ``dump_finetuned_predictions.py`` carry no
stay identifier. Their rows are in dataset order: the loader is built with
``shuffle=False``, ranks take contiguous shards, and the gathered tensors are
trimmed only at the tail, so row *i* is episode *i* of the extracted arrays.
That order is recorded in ``{fold}/{split}_ids.pkl`` as patient-episode IDs
(``patient_id * 1000 + episode_number``). The script asserts that the two
lengths agree and refuses to proceed otherwise, because a silent misalignment
would invalidate every number it prints.

Usage:
    python stratify_predictions_by_history.py \
        TransEHR2/configs/datasets/mimic4.yaml experiment2_text
    python stratify_predictions_by_history.py \
        TransEHR2/configs/datasets/mimic4.yaml experiment2_text \
        --split val --output-dir misc/stratified_carryforward
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import yaml

TASK = 'phenotype'
DEFAULT_AUDIT_CSV = os.path.join(
    'misc', 'historic_diagnosis_audit', 'strict',
    'historic_diagnosis_audit_per_episode.csv'
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_audit(audit_csv):
    """Read the per-stay audit table.

    Returns:
        Dict mapping patient-episode ID to a dict with `has_text` and the
        set of positive labels `named` in the earlier diagnosis text.
    """
    df = pd.read_csv(audit_csv)
    df = df[df['status'] == 'ok']

    records = {}
    for row in df.itertuples(index=False):
        # The episode key is .../<patient_id>/episode<N>.csv
        patient_id = os.path.basename(os.path.dirname(row.episode_key))
        if not patient_id.isdigit():
            continue
        episode_id = int(patient_id) * 1000 + int(row.episode_number)
        named = row.positive_labels_in_history
        records[episode_id] = {
            'has_text': bool(row.has_historical_dx_text),
            'named': set(named.split('|')) if isinstance(named, str) else set(),
        }
    return records


def discover_folds(data_dir, requested_folds=None):
    """Discover fold directories, matching extract_data.py's pattern."""
    if requested_folds:
        return requested_folds
    return sorted([
        item for item in os.listdir(data_dir)
        if re.match(r'fold\d+', item)
        and os.path.isdir(os.path.join(data_dir, item))
    ])


def load_fold(data_dir, model_dir, experiment_name, fold_name, split):
    """Load one fold's predictions, targets and episode IDs.

    Returns:
        Tuple of (scores, targets, episode_ids, phenotype_names), or None if
        this fold has no prediction file.

    Raises:
        ValueError: If the prediction rows and episode IDs disagree in
            number, which would mean the rows cannot be aligned to stays.
    """
    pred_path = os.path.join(
        model_dir, experiment_name, fold_name, TASK,
        f'{TASK}_{split}_finetuned_output.csv'
    )
    ids_path = os.path.join(data_dir, fold_name, f'{split}_ids.pkl')

    if not os.path.exists(pred_path):
        print(f"  {fold_name}: no predictions at {pred_path}", file=sys.stderr)
        return None
    if not os.path.exists(ids_path):
        print(f"  {fold_name}: no episode IDs at {ids_path}", file=sys.stderr)
        return None

    df = pd.read_csv(pred_path)
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    targ_cols = [c for c in df.columns if c.startswith('target_')]
    if not pred_cols:
        raise ValueError(
            f'{pred_path} has no pred_* columns; this script expects the '
            'phenotype task, whose columns are named after the diagnoses.'
        )
    names = [c[len('pred_'):] for c in pred_cols]
    if [c[len('target_'):] for c in targ_cols] != names:
        raise ValueError(f'Prediction and target columns disagree in {pred_path}')

    with open(ids_path, 'rb') as f:
        episode_ids = pickle.load(f)

    if len(episode_ids) != len(df):
        raise ValueError(
            f'{fold_name}: {len(df)} prediction rows but {len(episode_ids)} '
            f'episode IDs in {ids_path}. Rows cannot be aligned to stays; '
            'the predictions and the extracted arrays are out of step.'
        )

    return (df[pred_cols].to_numpy(dtype=float),
            df[targ_cols].to_numpy(dtype=float),
            list(episode_ids),
            names)


# ---------------------------------------------------------------------------
# Effect measures
# ---------------------------------------------------------------------------

def prob_higher(a, b):
    """Probability that a random draw from `a` exceeds one from `b`.

    Ties count as one half. This is the Mann-Whitney U statistic divided by
    the product of the group sizes, computed from midranks so that it is
    exact in the presence of ties.

    Args:
        a: Scores of the first group.
        b: Scores of the second group.

    Returns:
        The probability, or nan if either group is empty.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float('nan')
    pooled = np.concatenate([a, b])
    order = pooled.argsort(kind='mergesort')
    ranks = np.empty_like(pooled)
    ranks[order] = np.arange(1, pooled.size + 1, dtype=float)
    # Replace tied runs with their mean rank
    sorted_pooled = pooled[order]
    start = 0
    for i in range(1, sorted_pooled.size + 1):
        if i == sorted_pooled.size or sorted_pooled[i] != sorted_pooled[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    u = ranks[:a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def prevalence_matched_threshold(scores, n_positive):
    """Threshold that predicts exactly `n_positive` positives.

    Args:
        scores: All predicted scores for one diagnosis in one fold.
        n_positive: Number of observed positives.

    Returns:
        The threshold, or nan when it is not defined.
    """
    scores = np.asarray(scores, dtype=float)
    if n_positive <= 0 or n_positive > scores.size:
        return float('nan')
    return float(np.sort(scores)[-n_positive])


def summarise(values):
    """Return the mean and standard error of the mean, ignoring nans."""
    values = np.asarray([v for v in values if v == v], dtype=float)
    if values.size == 0:
        return float('nan'), float('nan')
    if values.size == 1:
        return float(values[0]), float('nan')
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


# ---------------------------------------------------------------------------
# Per-fold stratification
# ---------------------------------------------------------------------------

def stratify_fold(scores, targets, episode_ids, names, audit):
    """Split each diagnosis's true positives by carry-forward status.

    Args:
        scores: (n_stays, n_labels) predicted scores.
        targets: (n_stays, n_labels) binary labels.
        episode_ids: Patient-episode ID per row.
        names: Diagnosis name per column.
        audit: Mapping from :func:`load_audit`.

    Returns:
        Tuple of (per-diagnosis list of result dicts, coverage dict).
    """
    n_rows, n_labels = scores.shape
    has_text = np.zeros(n_rows, dtype=bool)
    known = np.zeros(n_rows, dtype=bool)
    # named_matrix[i, j] is True when diagnosis j is named in stay i's text
    named_matrix = np.zeros((n_rows, n_labels), dtype=bool)
    name_index = {name: j for j, name in enumerate(names)}

    for i, episode_id in enumerate(episode_ids):
        record = audit.get(episode_id)
        if record is None:
            continue
        known[i] = True
        has_text[i] = record['has_text']
        for label in record['named']:
            j = name_index.get(label)
            if j is not None:
                named_matrix[i, j] = True

    results = []
    for j, name in enumerate(names):
        positive = (targets[:, j] == 1) & known
        with_text = positive & has_text
        named = with_text & named_matrix[:, j]
        unnamed = with_text & ~named_matrix[:, j]
        no_text = positive & ~has_text

        threshold = prevalence_matched_threshold(
            scores[:, j], int(positive.sum())
        )
        recall = {}
        for key, mask in (('named', named), ('unnamed', unnamed),
                          ('no_text', no_text)):
            recall[key] = (float((scores[mask, j] >= threshold).mean())
                           if mask.any() and threshold == threshold
                           else float('nan'))

        results.append({
            'phenotype': name,
            'n_named': int(named.sum()),
            'n_unnamed': int(unnamed.sum()),
            'n_no_text': int(no_text.sum()),
            'mean_score_named': (float(scores[named, j].mean())
                                 if named.any() else float('nan')),
            'mean_score_unnamed': (float(scores[unnamed, j].mean())
                                   if unnamed.any() else float('nan')),
            'mean_score_no_text': (float(scores[no_text, j].mean())
                                   if no_text.any() else float('nan')),
            'prob_named_higher': prob_higher(scores[named, j],
                                             scores[unnamed, j]),
            'recall_named': recall['named'],
            'recall_unnamed': recall['unnamed'],
            'recall_no_text': recall['no_text'],
        })

    # Pooled over diagnoses: every (stay, diagnosis) true positive at once
    positive = (targets == 1) & known[:, None]
    with_text = positive & has_text[:, None]
    pooled_named = scores[with_text & named_matrix]
    pooled_unnamed = scores[with_text & ~named_matrix]
    pooled_no_text = scores[positive & ~has_text[:, None]]
    pooled = {
        'phenotype': 'All diagnoses pooled',
        'n_named': int(pooled_named.size),
        'n_unnamed': int(pooled_unnamed.size),
        'n_no_text': int(pooled_no_text.size),
        'mean_score_named': (float(pooled_named.mean())
                             if pooled_named.size else float('nan')),
        'mean_score_unnamed': (float(pooled_unnamed.mean())
                               if pooled_unnamed.size else float('nan')),
        'mean_score_no_text': (float(pooled_no_text.mean())
                               if pooled_no_text.size else float('nan')),
        'prob_named_higher': prob_higher(pooled_named, pooled_unnamed),
        'recall_named': float('nan'),
        'recall_unnamed': float('nan'),
        'recall_no_text': float('nan'),
    }

    coverage = {
        'n_rows': n_rows,
        'n_matched_to_audit': int(known.sum()),
        'n_with_text': int((has_text & known).sum()),
    }
    return results + [pooled], coverage


# ---------------------------------------------------------------------------

def aggregate(per_fold):
    """Average each diagnosis's per-fold statistics over folds.

    Args:
        per_fold: List of per-fold lists of result dicts.

    Returns:
        DataFrame with one row per diagnosis, plus the pooled row.
    """
    frames = pd.concat([pd.DataFrame(rows) for rows in per_fold])
    numeric = [c for c in frames.columns if c != 'phenotype']

    rows = []
    for name, group in frames.groupby('phenotype', sort=False):
        row = {'phenotype': name, 'n_folds': len(group)}
        for column in numeric:
            if column.startswith('n_'):
                row[column] = int(group[column].sum())
            else:
                mean, sem = summarise(group[column].tolist())
                row[column] = mean
                row[f'{column}_sem'] = sem
        row['score_difference'] = (
            row['mean_score_named'] - row['mean_score_unnamed']
        )
        row['recall_difference'] = row['recall_named'] - row['recall_unnamed']
        rows.append(row)

    result = pd.DataFrame(rows)
    pooled = result[result['phenotype'] == 'All diagnoses pooled']
    others = result[result['phenotype'] != 'All diagnoses pooled']
    others = others.sort_values('prob_named_higher', ascending=False)
    return pd.concat([pooled, others], ignore_index=True)


def print_report(result, experiment_name, split, coverages):
    total_rows = sum(c['n_rows'] for c in coverages)
    total_matched = sum(c['n_matched_to_audit'] for c in coverages)
    total_text = sum(c['n_with_text'] for c in coverages)

    print(f"\n{'='*100}")
    print(f'Carry-forward stratification: {experiment_name} ({split} split, '
          f'{len(coverages)} fold(s))')
    print(f"{'='*100}\n")
    print(f'  Stay-level rows across folds          {total_rows:,}')
    print(f'  Matched to the audit                  {total_matched:,}')
    print(f'  With earlier diagnosis text           {total_text:,}')
    print('\n  Among true positives in stays that have earlier diagnosis '
          'text, "named" means that text\n  already names the diagnosis. '
          'P(named>unnamed)=0.50 means the model does not distinguish them.\n')

    header = (f"  {'Diagnosis':<52s} {'n named':>8s} {'n unnam':>8s} "
              f"{'P(n>u)':>14s} {'Δ score':>16s} {'Δ recall':>16s}")
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for row in result.itertuples(index=False):
        prob = (f'{row.prob_named_higher:.3f}'
                if row.prob_named_higher == row.prob_named_higher else '—')
        if row.prob_named_higher_sem == row.prob_named_higher_sem:
            prob += f' ({row.prob_named_higher_sem:.3f})'
        score = (f'{row.score_difference:+.4f}'
                 if row.score_difference == row.score_difference else '—')
        rec = (f'{row.recall_difference:+.4f}'
               if row.recall_difference == row.recall_difference else '—')
        print(f'  {row.phenotype[:52]:<52s} {row.n_named:>8,d} '
              f'{row.n_unnamed:>8,d} {prob:>14s} {score:>16s} {rec:>16s}')
    print('\n  P(n>u) is shown as mean (SEM) over folds; Δ columns are '
          'differences of fold means.\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dataset_config',
                        help='YAML file specifying dataset parameters')
    parser.add_argument('experiment_name',
                        help='Experiment directory under --model-dir')
    parser.add_argument('--model-dir', default='./models',
                        help='Root directory of saved predictions')
    parser.add_argument('--audit-csv', default=DEFAULT_AUDIT_CSV,
                        help='Per-stay CSV from audit_historic_diagnoses.py')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to analyse (default: test)')
    parser.add_argument('--folds', nargs='*', default=None,
                        help='Specific folds (default: all)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for the output CSVs')
    args = parser.parse_args(argv)

    with open(args.dataset_config) as f:
        data_dir = yaml.safe_load(f)['DATA_DIR']

    audit = load_audit(args.audit_csv)
    print(f'Read {len(audit):,} audited stays from {args.audit_csv}')

    fold_names = discover_folds(data_dir, args.folds)
    if not fold_names:
        print('No fold directories found.', file=sys.stderr)
        return 1
    print(f'Found {len(fold_names)} fold(s): {fold_names}')

    per_fold, coverages, phenotype_names = [], [], None
    for fold_name in fold_names:
        loaded = load_fold(data_dir, args.model_dir, args.experiment_name,
                           fold_name, args.split)
        if loaded is None:
            continue
        scores, targets, episode_ids, names = loaded
        if phenotype_names is None:
            phenotype_names = names
        elif names != phenotype_names:
            raise ValueError(f'{fold_name}: diagnosis columns differ from '
                             'earlier folds')
        rows, coverage = stratify_fold(scores, targets, episode_ids, names,
                                       audit)
        per_fold.append(rows)
        coverages.append(coverage)
        print(f'  {fold_name}: {coverage["n_matched_to_audit"]:,}/'
              f'{coverage["n_rows"]:,} rows matched, '
              f'{coverage["n_with_text"]:,} with earlier text')

    if not per_fold:
        print('No folds had usable predictions.', file=sys.stderr)
        return 1

    result = aggregate(per_fold)
    print_report(result, args.experiment_name, args.split, coverages)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stem = f'{args.experiment_name}_{args.split}'
        by_label = os.path.join(args.output_dir, f'carryforward_{stem}.csv')
        result.to_csv(by_label, index=False)
        long_form = os.path.join(args.output_dir,
                                 f'carryforward_{stem}_per_fold.csv')
        pd.concat([pd.DataFrame(rows).assign(fold=fold)
                   for rows, fold in zip(per_fold, fold_names)]
                  ).to_csv(long_form, index=False)
        print(f'Wrote {by_label}\nWrote {long_form}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
