#!/usr/bin/env python
"""
Filter fold listfiles to retain only patient-episodes that have at least one
discharge summary in their historical records (records predating the current
ICU stay, i.e., Hours < 0 in the timeseries CSV).

Backs up original listfiles as {name}.unfiltered.csv, then overwrites with
filtered versions. Reports per-fold, per-partition filtering statistics.

Usage:
    python filter_listfiles_by_discharge_summary.py TransEHR2/configs/datasets/mimic4.yaml
    python filter_listfiles_by_discharge_summary.py TransEHR2/configs/datasets/mimic4.yaml --dry_run
    python filter_listfiles_by_discharge_summary.py TransEHR2/configs/datasets/mimic4.yaml --restore
"""

import argparse
import os
import re
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def has_historical_discharge_summary(episode_file_path):
    """Check whether an episode has discharge summary text in its historical records.

    Args:
        episode_file_path: Path to episodeX.csv. The corresponding timeseries
            file (episodeX_timeseries.csv) is read to check for discharge
            summaries at negative timestamps (pre-admission records).

    Returns:
        Tuple of (episode_file_path, bool) indicating whether the episode has
        at least one historical discharge summary.
    """
    ts_path = re.sub(r'\.csv$', '_timeseries.csv', episode_file_path)
    try:
        df = pd.read_csv(ts_path, usecols=['Hours', 'Discharge Summary'])
    except (FileNotFoundError, ValueError):
        return episode_file_path, False
    historical = df[df['Hours'] < 0]
    if historical.empty:
        return episode_file_path, False
    ds = historical['Discharge Summary'].dropna()
    if ds.empty:
        return episode_file_path, False
    has_text = (ds.astype(str).str.strip() != '').any()
    return episode_file_path, has_text


def check_episodes(episode_file_paths, n_workers=1):
    """Check which episodes have historical discharge summaries.

    Args:
        episode_file_paths: List of paths to episodeX.csv files.
        n_workers: Number of parallel workers.

    Returns:
        Set of episode file paths that have historical discharge summaries.
    """
    passing = set()
    unique_paths = list(set(episode_file_paths))

    if n_workers <= 1:
        for path in unique_paths:
            _, result = has_historical_discharge_summary(path)
            if result:
                passing.add(path)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(has_historical_discharge_summary, p): p
                for p in unique_paths
            }
            for future in as_completed(futures):
                path, result = future.result()
                if result:
                    passing.add(path)

    return passing


def backup_and_write(df, filepath, dry_run):
    """Back up original file and write filtered version.

    Args:
        df: Filtered DataFrame to write.
        filepath: Path to the listfile.
        dry_run: If True, skip writing.
    """
    if dry_run:
        return
    backup_path = re.sub(r'\.csv$', '.unfiltered.csv', filepath)
    if not os.path.exists(backup_path):
        os.rename(filepath, backup_path)
    else:
        # Backup already exists; just overwrite the filtered file
        pass
    df.to_csv(filepath, index=False)


def filter_fold_csv(fold_dir, fold_name, partition, passing_paths, dry_run):
    """Filter the fold dataset CSV.

    Returns:
        Tuple of (n_before, n_after) or None if file doesn't exist.
    """
    filepath = os.path.join(fold_dir, f'{fold_name}_{partition}.csv')
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    n_before = len(df)
    df = df[df['file_path'].isin(passing_paths)]
    n_after = len(df)
    backup_and_write(df, filepath, dry_run)
    return n_before, n_after


def filter_task_listfile(fold_dir, prefix, partition, passing_paths, dry_run):
    """Filter a task-specific listfile (phenotyping, mortality, or LOS).

    The 'stay' column contains paths to episodeX_timeseries.csv. These are
    converted to episodeX.csv paths to match against passing_paths.

    Returns:
        Tuple of (n_before, n_after) or None if file doesn't exist.
    """
    filepath = os.path.join(fold_dir, f'{prefix}_{partition}_listfile.csv')
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    n_before = len(df)
    # Convert timeseries paths to episode paths for matching
    episode_paths = df['stay'].apply(
        lambda p: re.sub(r'_timeseries\.csv$', '.csv', p)
    )
    df = df[episode_paths.isin(passing_paths)]
    n_after = len(df)
    backup_and_write(df, filepath, dry_run)
    return n_before, n_after


def restore_backups(data_dir, fold_names):
    """Restore .unfiltered.csv backups to their original filenames."""
    total_restored = 0
    for fold_name in fold_names:
        fold_dir = os.path.join(data_dir, fold_name)
        if not os.path.isdir(fold_dir):
            continue
        for filename in os.listdir(fold_dir):
            if filename.endswith('.unfiltered.csv'):
                original_name = filename.replace('.unfiltered.csv', '.csv')
                backup_path = os.path.join(fold_dir, filename)
                original_path = os.path.join(fold_dir, original_name)
                os.replace(backup_path, original_path)
                total_restored += 1
                print(f"  Restored {original_name}")
    return total_restored


def discover_folds(data_dir, requested_folds=None):
    """Discover fold directories, matching extract_data.py pattern."""
    if requested_folds:
        return requested_folds
    fold_names = sorted([
        item for item in os.listdir(data_dir)
        if re.match(r'fold\d+', item)
        and os.path.isdir(os.path.join(data_dir, item))
    ])
    return fold_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Filter fold listfiles to retain only episodes with historical "
            "discharge summaries (pre-admission text records)."
        )
    )
    parser.add_argument(
        'dataset_config',
        type=str,
        help="YAML file specifying dataset parameters (e.g., mimic4.yaml)"
    )
    parser.add_argument(
        '--folds',
        type=str,
        nargs='*',
        default=None,
        help="Specific folds to process (default: all folds)"
    )
    parser.add_argument(
        '--n_workers', '-w',
        type=int,
        default=1,
        help="Number of parallel workers for timeseries checking (default: 1)"
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help="Report statistics without modifying files"
    )
    parser.add_argument(
        '--restore',
        action='store_true',
        help="Restore .unfiltered.csv backups to undo filtering"
    )
    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)

    DATA_DIR = config['DATA_DIR']
    fold_names = discover_folds(DATA_DIR, args.folds)

    if not fold_names:
        print("No fold directories found.")
        exit(1)

    print(f"Found {len(fold_names)} fold(s): {fold_names}")

    if args.restore:
        print("\nRestoring backups...")
        n = restore_backups(DATA_DIR, fold_names)
        print(f"\nRestored {n} file(s).")
        exit(0)

    if args.dry_run:
        print("DRY RUN — no files will be modified.\n")

    TASK_PREFIXES = ['phenotyping', 'in_hospital_mortality', 'length_of_stay']
    total_before = 0
    total_after = 0

    for fold_name in fold_names:
        fold_dir = os.path.join(DATA_DIR, fold_name)

        print(f"\n{'='*60}")
        print(f"Processing {fold_name}")
        print(f"{'='*60}")

        for partition in ['train', 'test', 'val']:
            fold_csv = os.path.join(
                fold_dir, f'{fold_name}_{partition}.csv'
            )
            if not os.path.exists(fold_csv):
                if partition == 'val':
                    continue
                print(f"  WARNING: Missing {fold_csv}")
                continue

            # Read episode file paths from the fold CSV
            fold_df = pd.read_csv(fold_csv)
            episode_paths = fold_df['file_path'].tolist()

            print(f"\n  {partition}: checking {len(episode_paths)} episodes...")

            # Determine which episodes have historical discharge summaries
            passing = check_episodes(episode_paths, args.n_workers)

            # Filter fold CSV
            result = filter_fold_csv(
                fold_dir, fold_name, partition, passing, args.dry_run
            )
            if result:
                n_before, n_after = result
                pct = 100 * n_after / n_before if n_before > 0 else 0
                print(
                    f"  {fold_name}/{partition}: "
                    f"{n_before:,} -> {n_after:,} episodes "
                    f"({pct:.1f}% retained, {n_before - n_after:,} removed)"
                )
                total_before += n_before
                total_after += n_after

            # Filter task-specific listfiles
            for prefix in TASK_PREFIXES:
                result = filter_task_listfile(
                    fold_dir, prefix, partition, passing, args.dry_run
                )
                if result:
                    nb, na = result
                    pct = 100 * na / nb if nb > 0 else 0
                    print(
                        f"    {prefix}: "
                        f"{nb:,} -> {na:,} ({pct:.1f}% retained)"
                    )

    print(f"\n{'='*60}")
    if total_before > 0:
        pct = 100 * total_after / total_before
        print(
            f"Total: {total_before:,} -> {total_after:,} episodes "
            f"({pct:.1f}% retained)"
        )
    else:
        print("No episodes processed.")
    if args.dry_run:
        print("DRY RUN — no files were modified.")
    else:
        print("Filtering complete. Backups saved as *.unfiltered.csv")
    print(f"{'='*60}")
