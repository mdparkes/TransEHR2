#!/usr/bin/env python3
"""Verify that every feature in the tensorized dataset has at least one non-zero
indicator and value tensor across the training, validation, and test splits of fold0.

Usage:
    python test_data_integrity.py <dataset_config>

The script iterates through batches of 100 samples.  For each feature type
(numeric, categorical, text, event, static) it checks whether the indicator
tensor (or value tensor for static features) contains at least one non-zero
entry.  Iteration stops early once every feature has been confirmed or the
split is exhausted.
"""

import argparse
import os
import sys

import yaml

from functools import partial
from torch.utils.data import DataLoader

from TransEHR2.data.preprocessing import (VALUE_TYPES, collate_tensorized, load_dataset,
                                          partition_valued_feats)


BATCH_SIZE = 100


# Features whose absence is a property of MIMIC-IV, not a fault in the extraction. Troponin I
# has 670 rows in the whole of labevents against 529,212 for Troponin T, because BIDMC assays
# Troponin T, so no record need survive the cohort filters. Reported as expected rather than
# failing, so that a real gap still stands out.
EXPECTED_EMPTY = frozenset({'Troponin I'})


def check_split(split_path, feature_names):
    """Check a single data split for non-zero indicators/values.

    Args:
        split_path: Path to the tensorized split directory.
        feature_names: Dict mapping category -> list of feature names.

    Returns:
        Tuple of (verified, n_batches, n_samples) where `verified` maps a category key to the
        set of feature indices seen with at least one non-zero entry.
    """
    dataset = load_dataset(split_path)

    # The batch the model actually reads: the event stream sliced at the era boundary and the
    # text guard applied. history_len_steps must be the real width -- passing 0 means "no
    # history region", which makes the guard drop every text record and report text as
    # missing everywhere.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_tensorized,
                           history_len_steps=dataset.max_history_len_steps),
        num_workers=0,
    )

    verified = {}
    for stream in VALUE_TYPES:
        verified[f'{stream}_ind'] = set()
        verified[f'{stream}_val'] = set()
    verified.update({'text_ind': set(), 'text_emb': set(), 'event_ind': set(),
                     'static': set()})

    n_text = len(feature_names['text'])
    total_expected = (
        sum(2 * len(feature_names[stream]) for stream in VALUE_TYPES)
        + 2 * n_text + len(feature_names['event']) + len(feature_names['static'])
    )

    n_batches = 0
    for batch in loader:
        n_batches += 1

        for stream in VALUE_TYPES:
            block = batch['val_data'][stream]
            indicators, values = block['indicators'], block['values']
            for f in range(len(feature_names[stream])):
                if f not in verified[f'{stream}_ind'] and indicators[:, :, f].any():
                    verified[f'{stream}_ind'].add(f)
                if f not in verified[f'{stream}_val'] and values[f].any():
                    verified[f'{stream}_val'].add(f)

        text = batch['val_data'].get('text')
        if text is not None:
            for f in range(n_text):
                if f not in verified['text_ind'] and text['indicators'][:, :, f].any():
                    verified['text_ind'].add(f)
            # Embeddings arrive as one sparse COO-style block per feature and are densified
            # on device later, so a non-empty block is what shows the feature carries text.
            for f, sparse in enumerate(text.get('sparse_embeddings') or []):
                if f not in verified['text_emb'] and sparse['values'].numel():
                    verified['text_emb'].add(f)

        event_ind = batch['event_data']['indicators']
        for f in range(len(feature_names['event'])):
            if f not in verified['event_ind'] and event_ind[:, :, f].any():
                verified['event_ind'].add(f)

        static = batch['static_data']
        for f in range(len(feature_names['static'])):
            if f not in verified['static'] and static[:, f].any():
                verified['static'].add(f)

        if sum(len(seen) for seen in verified.values()) >= total_expected:
            break

    return verified, n_batches, len(dataset)


def report(split_name, verified, feature_names, n_batches, n_samples):
    """Print a report for a single split."""
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} split  ({n_samples} samples, {n_batches} batches)")
    print(f"{'='*60}")

    all_passed = True

    categories = []
    for stream in VALUE_TYPES:
        categories.append((f'{stream.capitalize()} (indicator)', f'{stream}_ind', stream))
        categories.append((f'{stream.capitalize()} (value)', f'{stream}_val', stream))
    categories += [
        ('Text (indicator)', 'text_ind', 'text'),
        ('Text (embedding)', 'text_emb', 'text'),
        ('Event (indicator)', 'event_ind', 'event'),
        ('Static (value)', 'static', 'static'),
    ]

    for label, key, feat_key in categories:
        names = feature_names[feat_key]
        n_total = len(names)
        n_verified = len(verified[key])
        missing_indices = set(range(n_total)) - verified[key]

        missing_names = [names[i] for i in sorted(missing_indices)]
        unexpected = [name for name in missing_names if name not in EXPECTED_EMPTY]
        expected = [name for name in missing_names if name in EXPECTED_EMPTY]

        if unexpected:
            all_passed = False
            print(f"\n  FAIL  {label}: {n_verified}/{n_total} passed")
            for name in unexpected:
                print(f"         - {name}")
        elif expected:
            print(f"  PASS  {label}: {n_verified}/{n_total} "
                  f"({len(expected)} expected empty)")
        else:
            print(f"  PASS  {label}: {n_verified}/{n_total}")
        for name in expected:
            print(f"         - {name} (expected empty, see EXPECTED_EMPTY)")

    if all_passed:
        print(f"\n  All features in {split_name} passed.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Test data integrity of tensorized fold0 splits'
    )
    parser.add_argument(
        'dataset_config', type=str,
        help='Path to the dataset YAML config file'
    )
    parser.add_argument(
        '--variable_properties', type=str, default=None,
        help='Path to the variable properties, overriding the config. The config records an '
             'absolute path, which does not resolve away from the cluster.'
    )
    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg['DATA_DIR']
    fold_dir = os.path.join(data_dir, 'fold0')

    if not os.path.isdir(fold_dir):
        print(f"ERROR: fold0 directory not found at {fold_dir}")
        sys.exit(1)

    # Name the value-stream features by repeating the partition the extraction performs.
    #
    # `_get_tensor_dimensions` groups VALUED_FEATS by the `type` in the variable properties and
    # writes one indicator tensor per type, preserving the config order within each type. The
    # config list is not grouped by type, so slicing it by the per-type counts attaches the
    # wrong name to every column past the first type boundary.
    properties_path = args.variable_properties or cfg['VARIABLE_PROPERTIES_PATH']
    with open(properties_path, 'r') as f:
        properties = yaml.safe_load(f)

    try:
        valued_by_type = partition_valued_feats(cfg['VALUED_FEATS'], properties)
    except ValueError as exc:
        print(f"ERROR: {exc} (properties from {properties_path})")
        sys.exit(1)

    feature_names = {
        **valued_by_type,
        'text': cfg.get('TEXT_FEATS', []) or [],
        'event': cfg['EVENT_FEATS'],
        'static': cfg['STATIC_FEATS'],
    }

    print("Data integrity test — fold0")
    print(f"Data directory: {fold_dir}")
    print("Features: " + ", ".join(
        f"{len(feature_names[key])} {key}"
        for key in list(VALUE_TYPES) + ['text', 'event', 'static']))

    overall_pass = True
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(fold_dir, split)
        if not os.path.isdir(split_path):
            if split == 'val':
                print(f"\n  (Skipping val split — directory not found)")
                continue
            else:
                print(f"\nERROR: {split} directory not found at {split_path}")
                sys.exit(1)

        verified, n_batches, n_samples = check_split(split_path, feature_names)
        passed = report(split, verified, feature_names, n_batches, n_samples)
        if not passed:
            overall_pass = False

    print(f"\n{'='*60}")
    if overall_pass:
        print("  OVERALL: ALL FEATURES PASSED in all splits.")
    else:
        print("  OVERALL: SOME FEATURES FAILED — see details above.")
    print(f"{'='*60}")

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
