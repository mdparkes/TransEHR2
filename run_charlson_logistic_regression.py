#!/usr/bin/env python3
"""Fit the Charlson comorbidity baseline for in-hospital mortality and dump its predictions.

A logistic regression on three features -- age at admission, sex, and the Charlson comorbidity
index of the most recent earlier hospital admission -- fitted per fold on that fold's training
split and applied to its validation and test splits. The predictions are written in the layout
`dump_finetuned_predictions.py` uses, so the existing reporter treats this arm as one more
column and the corrected resampled t test compares it against the in-stay-only model without
any special case:

    python report_results_tables.py --tasks mortality --cohorts charlson

Cohort and row order
--------------------
The cohort is the episode manifest `compute_charlson_index.py --write_cohort` writes: exactly
those episodes for which all three features exist. Both arms are given that one file -- this
one here, the in-stay-only control through `COHORT_EPISODES` in its experiment config -- and
both resolve it through the same `load_dataset(..., cohort_episodes=...)` call, which selects
rows by patient-episode ID in ascending array order. Row `i` of a prediction CSV is therefore
the same episode in both arms, which is what the paired test requires.

A manifest is used rather than a predicate over the arrays because membership depends on
something the arrays do not carry: whether the index could be computed from `diagnoses.csv`.
The array-side proxy `diagnosis_history` answers a different question -- whether the *text* of
an earlier admission's diagnoses survived extraction -- which neither arm reads, and which
excludes episodes whose codes are perfectly available.

Features
--------
Age and sex are read from the extracted `static_data` array rather than re-derived from the
source CSVs, so they are byte-for-byte the values the deep models receive. Note that the
extraction's `Age` is MIMIC-IV `anchor_age` carried onto every stay of a patient, not an age
recomputed at each admission. Sex is one-hot encoded against the most frequent category as
the reference, with categories too rare for a fold to estimate pooled into it -- see
`sex_encoding`. The Charlson index is joined by patient-episode ID from the table
`compute_charlson_index.py` writes.

The fit is unpenalized by default, which makes it plain maximum likelihood: with three
features there is no regularization strength to tune, and feature scaling cannot then affect
the predictions. The decision threshold is not chosen here -- the reporter calibrates it on
the validation split, as it does for every other arm.

Usage:
    python run_charlson_logistic_regression.py TransEHR2/configs/datasets/mimic4.yaml
    python run_charlson_logistic_regression.py TransEHR2/configs/datasets/mimic4.yaml \
        --charlson_csv misc/charlson/charlson_index.csv \
        --cohort_episodes misc/charlson/charlson_cohort.txt --model_dir models
    python run_charlson_logistic_regression.py TransEHR2/configs/datasets/mimic4.yaml \
        --class_weight balanced --penalty l2
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from TransEHR2.data.cohorts import load_episode_manifest
from TransEHR2.data.preprocessing import load_dataset
from TransEHR2.data.statics import decode_categorical, static_offsets

TASK = 'mortality'
SPLITS = ('train', 'val', 'test')
DEFAULT_EXPERIMENT = 'experiment19_charlson_logreg_charlsonsubset_rev'
DEFAULT_CHARLSON_CSV = os.path.join('misc', 'charlson', 'charlson_index.csv')
DEFAULT_COHORT = os.path.join('misc', 'charlson', 'charlson_cohort.txt')


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def sex_encoding(train_labels, min_count):
    """Choose the reference sex category and the ones to give an indicator.

    A category is encoded only if the training split holds at least `min_count` episodes of
    it; the rest are pooled into the reference. Without that floor an unpenalized fit can find
    a separating direction through a category with one or two members -- its coefficient
    diverges and those episodes get a prediction of essentially 0 or 1 on the strength of
    nothing. Pooling them into the reference costs a category no fold could estimate anyway.

    The reference is the most frequent category, ties broken alphabetically, so the encoding
    is a function of the data rather than of dictionary order.

    Args:
        train_labels: (n,) sex label strings from the training split.
        min_count: Fewest training episodes a category needs to be encoded.

    Returns:
        Tuple of (reference, indicators) where `indicators` is the sorted list of categories
        to encode, the reference excluded.
    """
    labels, counts = np.unique(np.asarray(train_labels), return_counts=True)
    order = sorted(zip(labels, counts), key=lambda pair: (-pair[1], pair[0]))
    reference = str(order[0][0])
    indicators = sorted(str(label) for label, count in order
                        if str(label) != reference and count >= min_count)
    return reference, indicators


def build_design_matrix(age, sex_labels, charlson, sex_indicators):
    """Assemble the design matrix from the three features.

    Args:
        age: (n,) age in years.
        sex_labels: (n,) sex label strings.
        charlson: (n,) Charlson comorbidity index.
        sex_indicators: Sex categories to encode, from `sex_encoding`. A label outside this
            list -- the reference, a pooled rare category, or one seen for the first time in
            the validation or test split -- contributes zeros, which places it in the
            reference bucket rather than dropping the episode.

    Returns:
        Tuple of (matrix, column_names) with the matrix of shape
        (n, 2 + len(sex_indicators)).
    """
    columns = [np.asarray(age, dtype=np.float64), np.asarray(charlson, dtype=np.float64)]
    names = ['age', 'charlson_index']
    for category in sex_indicators:
        columns.append((np.asarray(sex_labels) == category).astype(np.float64))
        names.append(f'sex_{category}')
    return np.column_stack(columns), names


def load_split(data_dir, fold, split, charlson, offsets, category_map, manifest,
               extracted_history_len_steps=None):
    """Load one fold-split's cohort episodes with their features and labels.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold directory name.
        split: Partition name.
        charlson: Series mapping patient-episode ID to Charlson index.
        offsets: Output of `static_offsets`.
        category_map: The `Gender` feature's category map.
        manifest: Path to the cohort's episode manifest, or the IDs themselves.
        extracted_history_len_steps: Width of the history region, for datasets written before
            the layout was recorded in metadata.

    Returns:
        Dict with `age`, `sex`, `charlson`, `mortality` and `episode_ids`, all in cohort row
        order, or None when the partition is not extracted.

    Raises:
        ValueError: If the episode IDs and the arrays are out of step, or if any cohort
            episode has no Charlson index.
    """
    base = os.path.join(data_dir, fold, split)
    if not os.path.exists(os.path.join(base, 'metadata.pkl')):
        return None

    # The cohort is applied by the loader, so `episode_indices` is exactly the row subset and
    # order that the in-stay-only control's inference loader produces for this partition.
    # The manifest is what the control arm is also given, so both select the same rows in the
    # same order and the reporter can pair them by position.
    dataset = load_dataset(base, cohort_episodes=manifest,
                           extracted_history_len_steps=extracted_history_len_steps)
    rows = dataset.episode_indices
    if rows is None:
        raise ValueError(
            f'{fold}/{split}: the loader applied no cohort, so the rows cannot be matched to '
            f'the control arm. The episode manifest should have selected a subset.'
        )

    ids_path = os.path.join(data_dir, fold, f'{split}_ids.pkl')
    with open(ids_path, 'rb') as handle:
        all_ids = pickle.load(handle)
    if len(all_ids) != dataset.n_extracted_episodes:
        raise ValueError(
            f'{fold}/{split}: {dataset.n_extracted_episodes} episodes in the arrays but '
            f'{len(all_ids)} ids in {ids_path}. The ids and the extracted arrays are out of '
            f'step, so features cannot be attributed to an episode.'
        )
    episode_ids = np.asarray(all_ids, dtype=np.int64)[rows]

    static = np.asarray(dataset.static_data)[rows]
    age = static[:, offsets['Age']].astype(np.float64)
    sex = decode_categorical(static[:, offsets['Gender']], category_map)

    missing = [int(episode_id) for episode_id in episode_ids
               if episode_id not in charlson.index]
    if missing:
        shown = ', '.join(str(i) for i in missing[:20])
        more = '' if len(missing) <= 20 else f', ... ({len(missing)} total)'
        raise ValueError(
            f'{fold}/{split}: {len(missing)} of {len(episode_ids)} cohort episodes have no '
            f'Charlson index: {shown}{more}. The manifest is supposed to name only episodes '
            f'that have one, so it and the index table are out of step -- rewrite it with '
            f'compute_charlson_index.py --write_cohort. Dropping the episodes is not an '
            f'option, because it would break the row correspondence with the control arm '
            f'that the paired test relies on.'
        )

    return {
        'age': age,
        'sex': sex,
        'charlson': charlson.reindex(episode_ids).to_numpy(dtype=np.float64),
        'mortality': np.asarray(dataset.mortality).astype(np.float64).ravel(),
        'episode_ids': episode_ids,
    }


# ---------------------------------------------------------------------------
# Fitting and output
# ---------------------------------------------------------------------------

def fit_fold(splits, sex_indicators, penalty, inverse_reg, class_weight, max_iter):
    """Fit the model on the training split and predict every available split.

    Args:
        splits: Mapping from split name to the output of `load_split`.
        sex_indicators: Sex categories to encode, from `sex_encoding`.
        penalty: `'l2'` or None.
        inverse_reg: sklearn's `C`; ignored when `penalty` is None.
        class_weight: `'balanced'` or None.
        max_iter: Solver iteration cap.

    Returns:
        Tuple of (predictions, model, column_names) where `predictions` maps split name to
        (probabilities, targets).

    Raises:
        ValueError: If the training split has only one mortality outcome, which leaves the
            model undefined.
    """
    train = splits['train']
    design, names = build_design_matrix(train['age'], train['sex'], train['charlson'],
                                        sex_indicators)
    targets = train['mortality']
    if len(np.unique(targets)) < 2:
        raise ValueError(
            f'the training split has a single mortality outcome '
            f'({int(targets[0]) if len(targets) else "no"}), so a logistic regression is not '
            f'identifiable on it.'
        )

    model = LogisticRegression(
        penalty=penalty, C=inverse_reg, class_weight=class_weight,
        max_iter=max_iter, solver='lbfgs',
    )
    model.fit(design, targets)

    predictions = {}
    for split, data in splits.items():
        if data is None:
            continue
        matrix, _ = build_design_matrix(data['age'], data['sex'], data['charlson'],
                                        sex_indicators)
        predictions[split] = (model.predict_proba(matrix)[:, 1], data['mortality'])
    return predictions, model, names


def write_predictions(model_dir, experiment_name, fold, split, probabilities, targets):
    """Write one prediction CSV in the layout the reporter reads.

    Args:
        model_dir: Directory holding one subdirectory per experiment.
        experiment_name: Experiment directory name.
        fold: Fold directory name.
        split: Partition name.
        probabilities: (n,) predicted probability of in-hospital death.
        targets: (n,) observed outcome.

    Returns:
        The path written.
    """
    directory = os.path.join(model_dir, experiment_name, fold, TASK)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{TASK}_{split}_finetuned_output.csv')
    pd.DataFrame({'prediction': np.asarray(probabilities, dtype=np.float64),
                  'target': np.asarray(targets, dtype=np.float64)}).to_csv(path, index=False)
    return path


def coefficient_table(model, names):
    """The fitted coefficients as a table, with odds ratios.

    Args:
        model: A fitted `LogisticRegression`.
        names: Column names of the design matrix, in order.

    Returns:
        A DataFrame with one row per term, the intercept first.
    """
    coefficients = np.ravel(model.coef_)
    rows = [{'term': 'intercept', 'coefficient': float(model.intercept_[0]),
             'odds_ratio': float(np.exp(model.intercept_[0]))}]
    rows.extend({'term': name, 'coefficient': float(value),
                 'odds_ratio': float(np.exp(value))}
                for name, value in zip(names, coefficients))
    return pd.DataFrame(rows)


def get_fold_names(data_dir, exclude=('fold0',)):
    """Fold directories to fit, excluding those reserved for tuning."""
    folds = [item for item in os.listdir(data_dir)
             if item not in exclude and re.match(r'fold\d+', item)
             and os.path.isdir(os.path.join(data_dir, item))]
    folds.sort(key=lambda name: int(name[4:]))
    return folds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dataset_config', help='YAML file specifying dataset parameters')
    parser.add_argument('--data_dir', default=None,
                        help='Override DATA_DIR from the dataset config')
    parser.add_argument('--charlson_csv', default=DEFAULT_CHARLSON_CSV,
                        help=f'Table written by compute_charlson_index.py '
                             f'(default: {DEFAULT_CHARLSON_CSV})')
    parser.add_argument('--cohort_episodes', default=DEFAULT_COHORT,
                        help=f'Episode manifest written by compute_charlson_index.py '
                             f'--write_cohort. Must be the same file the control arm names in '
                             f'COHORT_EPISODES (default: {DEFAULT_COHORT})')
    parser.add_argument('--model_dir', default='models',
                        help='Directory holding one subdirectory per experiment '
                             '(default: models)')
    parser.add_argument('--experiment_name', default=DEFAULT_EXPERIMENT,
                        help=f'Experiment directory to write into '
                             f'(default: {DEFAULT_EXPERIMENT})')
    parser.add_argument('--folds', nargs='+', default=None,
                        help='Folds to fit (default: every fold except fold0)')
    parser.add_argument('--penalty', choices=('none', 'l2'), default='none',
                        help='Regularization; "none" makes the fit plain maximum likelihood '
                             '(default: none)')
    parser.add_argument('-C', '--inverse_reg', type=float, default=1.0,
                        help='Inverse regularization strength, with --penalty l2 (default: 1)')
    parser.add_argument('--class_weight', choices=('none', 'balanced'), default='none',
                        help='Class weighting; the reporter calibrates the decision threshold '
                             'either way (default: none)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Solver iteration cap (default: 1000)')
    parser.add_argument('--min_category_count', type=int, default=10,
                        help='Fewest training episodes a sex category needs before it gets '
                             'its own indicator; rarer ones are pooled into the reference '
                             '(default: 10)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Fit and report without writing any predictions')
    args = parser.parse_args(argv)

    with open(args.dataset_config) as handle:
        dataset_config = yaml.safe_load(handle)
    data_dir = args.data_dir or dataset_config['DATA_DIR']
    with open(dataset_config['VARIABLE_PROPERTIES_PATH']) as handle:
        variable_properties = yaml.safe_load(handle)

    static_feats = dataset_config['STATIC_FEATS']
    for required in ('Age', 'Gender'):
        if required not in static_feats:
            raise SystemExit(
                f'{args.dataset_config} does not list {required!r} in STATIC_FEATS, so the '
                f'extracted arrays do not carry it and the model cannot be fitted.'
            )
    offsets = static_offsets(variable_properties, static_feats,
                             dataset_config.get('MAX_TOKEN_LENGTH', 0))
    category_map = variable_properties['Gender'].get('category_map', {})

    if not os.path.exists(args.cohort_episodes):
        raise SystemExit(
            f'{args.cohort_episodes} does not exist. Write it with compute_charlson_index.py '
            f'--write_cohort; it is what puts this arm and the control on the same episodes.'
        )
    cohort = load_episode_manifest(args.cohort_episodes)
    print(f'Cohort: {len(cohort)} episodes from {args.cohort_episodes}')

    charlson_frame = pd.read_csv(args.charlson_csv)
    charlson = charlson_frame.set_index('episode_id')['charlson_index']
    if charlson.index.has_duplicates:
        duplicated = charlson.index[charlson.index.duplicated()].unique()
        raise SystemExit(
            f'{args.charlson_csv} has {len(duplicated)} repeated episode ids, so an episode '
            f'would take an arbitrary one of several indices.'
        )
    print(f'Charlson indices: {len(charlson)} episodes, '
          f'mean {charlson.mean():.2f}, range {charlson.min()}-{charlson.max()}')

    fold_names = args.folds or get_fold_names(data_dir)
    if not fold_names:
        raise SystemExit(f'No fold directories found in {data_dir}')

    penalty = None if args.penalty == 'none' else args.penalty
    class_weight = None if args.class_weight == 'none' else args.class_weight

    coefficient_tables = []
    for fold in fold_names:
        print(f'\n=== {fold} ===')
        splits = {}
        for split in SPLITS:
            splits[split] = load_split(data_dir, fold, split, charlson, offsets,
                                       category_map, args.cohort_episodes,
                                       dataset_config.get('MAX_HISTORY_LEN_STEPS'))
        if splits.get('train') is None:
            print('  no extracted train partition; skipped')
            continue

        for split, data in splits.items():
            if data is not None:
                print(f'  {split:5} {len(data["mortality"]):6} episodes, '
                      f'{int(data["mortality"].sum()):5} deaths '
                      f'({100.0 * data["mortality"].mean():.1f}%), '
                      f'mean Charlson {data["charlson"].mean():.2f}')

        reference, sex_indicators = sex_encoding(splits['train']['sex'],
                                                 args.min_category_count)
        pooled = sorted(set(splits['train']['sex'].tolist())
                        - {reference} - set(sex_indicators))
        print(f'  sex: reference {reference!r}, indicators {sex_indicators}'
              + (f', pooled into the reference {pooled}' if pooled else ''))

        predictions, model, names = fit_fold(splits, sex_indicators, penalty,
                                             args.inverse_reg, class_weight, args.max_iter)
        table = coefficient_table(model, names)
        table.insert(0, 'fold', fold)
        coefficient_tables.append(table)
        print('  coefficients: ' + ', '.join(
            f'{row.term}={row.coefficient:+.4f}' for row in table.itertuples(index=False)
        ))

        if args.dry_run:
            continue
        for split, (probabilities, targets) in predictions.items():
            path = write_predictions(args.model_dir, args.experiment_name, fold, split,
                                     probabilities, targets)
            print(f'  -> {path}')

    if args.dry_run:
        print('\nDry run: nothing written.')
        return 0
    if not coefficient_tables:
        print('\nNo folds were fitted.')
        return 1

    coefficients_path = os.path.join(args.model_dir, args.experiment_name,
                                     'charlson_coefficients.csv')
    pd.concat(coefficient_tables, ignore_index=True).to_csv(coefficients_path, index=False)
    print(f'\nCoefficients written to {coefficients_path}')
    print(f'Predictions written under {os.path.join(args.model_dir, args.experiment_name)}/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
