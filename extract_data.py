#!/usr/bin/env python
"""Extract the cohort's episodes into ``data/extracted/`` (section 4).

Runs **once for the cohort**, not once per fold and not once per
partition. The row order is ``labels.csv``'s -- patient directories
lexicographically, then ``STAY_INDEX`` -- and section 3's folds are
``int64`` row indices into the arrays this writes, so the row count is
fixed by ``labels.csv`` and no episode may be filtered out.

Inputs, all named by the dataset config:

    {DATA_DIR}/root/{PATID}/{stays,timeseries,drugs}.csv   Stage B
    {DATA_DIR}/labels.csv                                  Stage B'
    {DATA_DIR}/fold{i}/fold{i}_train_rows.npy              split.py
    VARIABLE_PROPERTIES_PATH, CLINVEC_PATH

Output is ``{DATA_DIR}/extracted/``, whose contract is section 4.4:

    Dense, per episode (n episodes, T = MAX_EPISODE_LEN_STEPS):
        val_times.npy                (n, T)      int32, minutes <= 0
        val_masks.npy                (n, T)      float32
        val_numeric_indicators.npy   (n, T, 94)  float32
        val_numeric_values_{i}.npy   (n, T, 1)   float32
        val_categorical_indicators.npy (n, T, 37) float32
        val_categorical_values_{i}.npy (n, T)    int16 index, -1 sentinel
        val_ordinal_indicators.npy   (n, T, 16)  float32
        val_ordinal_values_{i}.npy   (n, T)      int16 index
        val_text_indicators.npy      (n, T, 1)   float32
        val_drug_indicators.npy      (n, T, 1)   float32
        event_indicators.npy         (n, T, 1)   float32
        event_times.npy              (n, T)      int32
        event_masks.npy              (n, T)      float32
        index_times.npy              (n,)        datetime64[ns]
        time_to_event.npy            (n,)        float32
        event_type.npy               (n,)        int8

    Sparse text and drugs, CSR over the non-empty timesteps:
        text_offsets_{i}.npy   (n+1,)          int64
        text_timesteps_{i}.npy (n_nonempty,)   int32
        text_values_{i}.npy    (n_nonempty,)   int32 row in the table
        drug_offsets_{i}.npy   (n+1,)          int64
        drug_timesteps_{i}.npy (n_nonempty,)   int32
        drug_values_{i}.npy    (n_nonempty,30) int32, right-padded with V
        drug_doses_{i}.npy     (n_nonempty,30) float32, 0 on pad
        drug_masks_{i}.npy     (n_nonempty,30) float32, 1 real 0 pad

    Metadata:
        metadata.pkl, episode_ids.pkl, text_strings.pkl,
        summary_statistics_fold{i}.npz

There is **no** ``static_data.npy``: ``STATIC_FEATS`` is empty under
section A.3, so the array would be ``(n, 0)``.

``text_strings.pkl`` is the unique-string table's key order, assigned here
because section 9 makes neither C1 nor C4 depend on the other; C4's
``embed_text.py`` embeds that list *in that order*, which is what makes
``text_values`` valid indices into ``text_embeddings.npy``.
"""

import argparse
import numpy as np
import os
import re
import sys
import yaml

from TransEHR2.data.datareaders import EHRDataReader
from TransEHR2.data.preprocessing import (
    _bucket_valued_feats, check_feature_contract, extract_data
)


def find_fold_train_rows(data_dir: str) -> dict:
    """Collect ``fold{i}_train_rows.npy`` from each ``fold{i}/``.

    One set of standardization statistics per fold, computed over that
    fold's *training* rows alone -- statistics over the whole cohort would
    leak val and test into the scaling (section 5).
    """
    fold_train_rows = {}
    if not os.path.isdir(data_dir):
        return fold_train_rows
    for item in sorted(os.listdir(data_dir)):
        if not re.fullmatch(r'fold\d+', item):
            continue
        rows_path = os.path.join(data_dir, item, f'{item}_train_rows.npy')
        if os.path.exists(rows_path):
            fold_train_rows[item] = np.load(rows_path)
        else:
            print(f"  {item}/ has no {item}_train_rows.npy; no "
                  f"summary_statistics_{item}.npz will be written.")
    return fold_train_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the cohort into data/extracted/ (section 4)"
    )
    parser.add_argument(
        'dataset_config',
        type=str,
        help="YAML file specifying dataset parameters (RMT23345.yaml)"
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default=None,
        help="Override the config's DATA_DIR, e.g. to run over a fixture"
    )
    parser.add_argument(
        '--n_examples', '-n',
        type=int,
        default=None,
        help="Extract only the first N rows of labels.csv (debugging)"
    )
    parser.add_argument(
        '--n_workers', '-w',
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    args = parser.parse_args(argv)

    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)

    DATA_DIR = args.data_dir or config['DATA_DIR']
    VAR_PROPERTIES_PATH = config['VARIABLE_PROPERTIES_PATH']
    CLINVEC_PATH = config.get('CLINVEC_PATH')
    VALUED_FEATS = config['VALUED_FEATS']
    EVENT_FEATS = config['EVENT_FEATS']
    TEXT_FEATS = config.get('TEXT_FEATS') or []
    DRUG_FEATS = config.get('DRUG_FEATS') or []
    STATIC_FEATS = config.get('STATIC_FEATS') or []
    MAX_EPISODE_LEN_STEPS = config.get('MAX_EPISODE_LEN_STEPS', 500)
    MIN_EPISODE_LEN_STEPS = config.get('MIN_EPISODE_LEN_STEPS', 1)

    with open(VAR_PROPERTIES_PATH, 'r') as f:
        var_properties = yaml.safe_load(f)

    # Invariant 12, before a single patient file is opened (section 6).
    # The category_map clause used to fire mid-extraction, on real data,
    # hours in (section A.3).
    failures = check_feature_contract(config, var_properties)
    if failures:
        print("FAIL -- the feature contract does not hold (invariant 12):")
        for failure in failures:
            print(f"  {failure}")
        return 1

    numeric_feats, categorical_feats, ordinal_feats, _ = (
        _bucket_valued_feats(
            VALUED_FEATS, TEXT_FEATS + DRUG_FEATS, var_properties
        )
    )
    print(f"Feature contract holds: {len(numeric_feats)} numeric, "
          f"{len(categorical_feats)} categorical, "
          f"{len(ordinal_feats)} ordinal.")

    # Every categorical, ordinal and text column is read as a string.
    # category_map is keyed on the YAML's values and BLDUA's levels are
    # numeric-looking strings, so a column pandas typed as numeric would
    # miss every one of them, silently and per patient (section 5).
    string_feats = categorical_feats + ordinal_feats + TEXT_FEATS

    labels_path = os.path.join(DATA_DIR, 'labels.csv')
    root_dir = os.path.join(DATA_DIR, 'root')
    for path in (labels_path, root_dir):
        if not os.path.exists(path):
            print(f"{path} not found. Run IBDdataprep's build_root.py and "
                  f"build_labels.py first.")
            return 1

    reader = EHRDataReader(
        labels_path=labels_path,
        root_dir=root_dir,
        valued_feats=VALUED_FEATS,
        event_feats=EVENT_FEATS,
        text_feats=TEXT_FEATS,
        drug_feats=DRUG_FEATS,
        static_feats=STATIC_FEATS,
        string_feats=string_feats,
        n_examples=args.n_examples
    )

    fold_train_rows = find_fold_train_rows(DATA_DIR)
    if fold_train_rows:
        print(f"Standardization statistics for: "
              f"{sorted(fold_train_rows)}")
    else:
        print("No fold{i}/fold{i}_train_rows.npy found; no "
              "standardization statistics will be written.")
    sys.stdout.flush()

    extract_data(
        reader=reader,
        output_dir=DATA_DIR,
        var_properties_path=VAR_PROPERTIES_PATH,
        max_episode_len_steps=MAX_EPISODE_LEN_STEPS,
        clinvec_path=CLINVEC_PATH,
        min_episode_len_steps=MIN_EPISODE_LEN_STEPS,
        n_workers=args.n_workers,
        fold_train_rows=fold_train_rows
    )

    print("\nExtraction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
