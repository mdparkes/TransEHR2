#!/usr/bin/env python
"""
Extract MIMIC ICU stay timeseries data from CSV files into a tensorized format optimized for 
fast data loading during model training.

This function reads patient episode data using a MIMICDataReader, applies filtering criteria,
processes the data into pre-allocated numpy arrays, standardizes numeric features, and saves 
the result as a directory of memory-mappable .npy files. The output format is designed for 
efficient multi-worker DataLoader access with minimal memory overhead.

The extraction pipeline operates in two passes:
    1. **Filtering pass**: Episodes are processed in parallel to apply inclusion criteria 
        (minimum length, minimum timesteps) and extract raw data.
    2. **Insertion pass**: Surviving episodes are inserted into pre-allocated arrays and 
        text features are converted to sparse storage format.

Output Structure:
    The function creates a directory at `{output_dir}/{suffix}/` containing:
    
    * Dense arrays (one .npy file each):
        - `val_times.npy`: (n_episodes, max_ts_len) float32 - Timestamps in hours
        - `val_masks.npy`: (n_episodes, max_ts_len) float32 - Valid timestep indicators
        - `val_numeric_indicators.npy`: (n_episodes, max_ts_len, n_numeric_feats) float32
        - `val_categorical_indicators.npy`: (n_episodes, max_ts_len, n_cat_feats) float32
        - `val_text_indicators.npy`: (n_episodes, max_ts_len, n_text_feats) float32
        - `event_times.npy`: (n_episodes, max_ts_len) float32
        - `event_masks.npy`: (n_episodes, max_ts_len) float32
        - `event_indicators.npy`: (n_episodes, max_ts_len, n_event_feats) float32
        - `static_data.npy`: (n_episodes, static_total_dim) float32
        - `mortality.npy`: (n_episodes,) float32
        - `length_of_stay.npy`: (n_episodes,) float32
        - `phenotype.npy`: (n_episodes, n_phenotypes) float32
    
    * Per-feature arrays:
        - `val_numeric_values_{i}.npy`: (n_episodes, max_ts_len, feat_dim) float32
        - `val_categorical_values_{i}.npy`: (n_episodes, max_ts_len, feat_dim) int64
    
    * Sparse text arrays (CSR-style storage for memory efficiency):
        - `val_text_offsets_{i}.npy`: (n_episodes + 1,) int64 - CSR row pointers
        - `val_text_values_{i}.npy`: (n_non_empty, token_len) int64 - Token IDs
        - `val_text_masks_{i}.npy`: (n_non_empty, token_len) float32 - Attention masks
        - `val_text_timesteps_{i}.npy`: (n_non_empty,) int32 - Original timestep indices
    
    * Metadata:
        - `metadata.pkl`: Dictionary containing dimension information for reconstruction
    
    Additionally creates:
        - `{suffix}_ids.pkl`: List of patient episode IDs that passed filtering
        - `summary_statistics_train.npz`: (train only) Standardization parameters.
"""

import argparse
import os
import re
import yaml

from TransEHR2.data.datareaders import MIMICDataReader
from TransEHR2.data.preprocessing import extract_mimic


def check_for_train_test_listfiles(fold_dir: str, fold_name: str) -> None:
    """Verify that required listfiles exist for a fold."""
    for partition in ['train', 'test']:
        dataset_listfile = os.path.join(fold_dir, f'{fold_name}_{partition}.csv')
        phenotypes_listfile = os.path.join(fold_dir, f'phenotyping_{partition}_listfile.csv')
        if not os.path.exists(dataset_listfile):
            raise FileNotFoundError(f"Missing listfile {dataset_listfile}")
        if not os.path.exists(phenotypes_listfile):
            raise FileNotFoundError(f"Missing listfile {phenotypes_listfile}")


def skip_validation(fold_dir: str, fold_name: str) -> bool:
    """Check if validation listfiles are missing (validation is optional)."""
    dataset_listfile = os.path.join(fold_dir, f'{fold_name}_val.csv')
    phenotypes_listfile = os.path.join(fold_dir, f'phenotyping_val_listfile.csv')
    skip = False
    if not os.path.exists(dataset_listfile):
        print(f"  Missing validation dataset listfile {dataset_listfile}.")
        skip = True
    if not os.path.exists(phenotypes_listfile):
        print(f"  Missing validation phenotypes listfile {phenotypes_listfile}.")
        skip = True
    if skip:
        print("  Skipping validation set.\n")
    return skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MIMIC data directly into tensorized format for fast training"
    )
    parser.add_argument(
        'dataset_config',
        type=str,
        help="YAML file specifying dataset parameters (e.g., mimic4.yaml)"
    )
    parser.add_argument(
        '--n_examples', '-n',
        type=int,
        default=None,
        help="Number of examples to process per partition (for debugging)"
    )
    parser.add_argument(
        '--n_workers', '-w',
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    parser.add_argument(
        '--folds',
        type=str,
        nargs='*',
        default=None,
        help="Specific folds to process (default: all folds)"
    )
    args = parser.parse_args()
    
    # Load dataset configuration
    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    
    DATA_DIR = config['DATA_DIR']
    VAR_PROPERTIES_PATH = config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = config['VALUED_FEATS']
    EVENT_FEATS = config['EVENT_FEATS']
    TEXT_FEATS = config.get('TEXT_FEATS', None)
    STATIC_FEATS = config['STATIC_FEATS']
    MAX_EPISODE_LEN_STEPS = config.get('MAX_EPISODE_LEN_STEPS', 500)
    MAX_HISTORY_LEN_STEPS = config.get('MAX_HISTORY_LEN_STEPS', 0)
    MIN_EPISODE_LEN_STEPS = config.get('MIN_EPISODE_LEN_STEPS', 10)
    MIN_EPISODE_LEN_HOURS = config.get('MIN_EPISODE_LEN_HOURS', 48)
    MAX_EPISODE_LEN_HOURS = config.get('MAX_EPISODE_LEN_HOURS', 48)
    
    # Find fold directories
    if args.folds:
        fold_names = args.folds
    else:
        fold_names = []
        for item in os.listdir(DATA_DIR):
            if re.match(r'fold\d+', item) and os.path.isdir(os.path.join(DATA_DIR, item)):
                fold_names.append(item)
        fold_names.sort()
    
    print(f"Found {len(fold_names)} fold(s) to process: {fold_names}")
    print(f"Using {args.n_workers} worker(s)\n")
    
    # Process each fold
    for fold_name in fold_names:
        fold_dir = os.path.join(DATA_DIR, fold_name)
        
        print(f"{'='*60}")
        print(f"Processing {fold_name}")
        print(f"{'='*60}")
        
        check_for_train_test_listfiles(fold_dir, fold_name)
        
        for partition in ['train', 'test', 'val']:
            if partition == 'val' and skip_validation(fold_dir, fold_name):
                continue
            
            dataset_listfile = os.path.join(fold_dir, f'{fold_name}_{partition}.csv')
            phenotypes_listfile = os.path.join(fold_dir, f'phenotyping_{partition}_listfile.csv')
            
            print(f"\nInitializing datareader for {fold_name}, {partition} set...")
            
            reader = MIMICDataReader(
                dataset_listfile=dataset_listfile,
                phenotypes_listfile=phenotypes_listfile,
                valued_feats=VALUED_FEATS,
                event_feats=EVENT_FEATS,
                static_feats=STATIC_FEATS,
                text_feats=TEXT_FEATS,
                prediction_task='all',
                n_examples=args.n_examples
            )
            
            extract_mimic(
                reader=reader,
                suffix=partition,
                output_dir=fold_dir,
                var_properties_path=VAR_PROPERTIES_PATH,
                max_episode_len_steps=MAX_EPISODE_LEN_STEPS,
                max_history_len_steps=MAX_HISTORY_LEN_STEPS,
                min_episode_len_steps=MIN_EPISODE_LEN_STEPS,
                min_episode_len_hours=MIN_EPISODE_LEN_HOURS,
                max_episode_len_hours=MAX_EPISODE_LEN_HOURS,
                n_workers=args.n_workers
            )
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")