"""
Compute evaluation metrics from finetuned TransEHR2 prediction CSVs.

Reads the CSV files produced by ``dump_finetuned_predictions.py`` and
calculates task-specific evaluation metrics for each cross-validation
fold.  Results are written to one YAML file per data split at
``{model_dir}/{experiment_name}/{split}_evaluation.yaml``, with each
metric stored as a list of per-fold values.

Usage:
    python evaluate_finetuned_predictions.py <experiment_name> \
        [--model_dir ./models] [--threshold 0.5]
"""

import argparse
import glob
import numpy as np
import os
import re
import yaml

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score
)
from typing import Dict, List, Optional


TASKS = ['mortality', 'length_of_stay', 'phenotype']
SPLITS = ['train', 'val', 'test']


# ------------------------------------------------------------------
# Concordance index
# ------------------------------------------------------------------

def concordance_index(predicted: np.ndarray,
                      observed: np.ndarray) -> float:
    """Compute Harrell's concordance index for non-censored data.

    For every pair of observations (i, j) where observed[i] != observed[j],
    the pair is *concordant* when the ordering of the predicted values
    agrees with the ordering of the observed values.  Ties in
    predictions count as 0.5 concordant.

    Args:
        predicted: Predicted values, shape (n,).
        observed: Observed (true) values, shape (n,).

    Returns:
        Concordance index in [0, 1].  Returns ``nan`` if there are
        fewer than two observations or no valid pairs.
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    observed = np.asarray(observed, dtype=np.float64).ravel()

    n = len(predicted)
    if n < 2:
        return float('nan')

    concordant = 0.0
    total = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if observed[i] == observed[j]:
                continue
            total += 1.0
            if observed[i] > observed[j]:
                if predicted[i] > predicted[j]:
                    concordant += 1.0
                elif predicted[i] == predicted[j]:
                    concordant += 0.5
            else:
                if predicted[i] < predicted[j]:
                    concordant += 1.0
                elif predicted[i] == predicted[j]:
                    concordant += 0.5

    if total == 0.0:
        return float('nan')

    return concordant / total


# ------------------------------------------------------------------
# Binary classification metrics
# ------------------------------------------------------------------

def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute a full set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_prob: Predicted probabilities, shape (n,).
        threshold: Decision threshold for converting probabilities
            to hard class labels.

    Returns:
        Dictionary of metric names to float values.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    metrics: Dict[str, float] = {}

    metrics['accuracy'] = float(accuracy_score(y_true_int, y_pred))
    metrics['f1'] = float(f1_score(y_true_int, y_pred, zero_division=0))
    metrics['precision'] = float(
        precision_score(y_true_int, y_pred, zero_division=0)
    )
    metrics['recall_sensitivity'] = float(
        recall_score(y_true_int, y_pred, zero_division=0)
    )

    # AUROC / AUPRC require both classes to be present
    if len(np.unique(y_true_int)) > 1:
        metrics['auroc'] = float(roc_auc_score(y_true_int, y_prob))
        metrics['auprc'] = float(
            average_precision_score(y_true_int, y_prob)
        )
    else:
        metrics['auroc'] = float('nan')
        metrics['auprc'] = float('nan')

    # Confusion-matrix derived rates
    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics['specificity'] = float(specificity)
    metrics['false_positive_rate'] = float(1.0 - specificity)
    metrics['false_negative_rate'] = float(1.0 - sensitivity)
    metrics['ppv'] = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    metrics['npv'] = float(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
    metrics['false_discovery_rate'] = float(
        fp / (fp + tp) if (fp + tp) > 0 else 0.0
    )
    metrics['prevalence'] = float(np.mean(y_true))

    return metrics


# ------------------------------------------------------------------
# Task-level metric functions
# ------------------------------------------------------------------

def compute_mortality_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute mortality (binary classification) metrics.

    Args:
        predictions: Predicted probabilities, shape (n, 1) or (n,).
        targets: Binary targets, shape (n, 1) or (n,).
        threshold: Classification threshold.

    Returns:
        Dictionary of metric name to value.
    """
    return compute_binary_metrics(
        targets.ravel(), predictions.ravel(), threshold
    )


def compute_los_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """Compute length-of-stay (regression) metrics.

    Args:
        predictions: Predicted LOS values, shape (n, 1) or (n,).
        targets: True LOS values, shape (n, 1) or (n,).

    Returns:
        Dictionary with ``mean_absolute_error`` and
        ``concordance_index``.
    """
    pred = predictions.ravel()
    targ = targets.ravel()

    return {
        'mean_absolute_error': float(mean_absolute_error(targ, pred)),
        'concordance_index': float(concordance_index(pred, targ))
    }


def compute_phenotype_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute phenotype (multi-label binary) metrics.

    Each binary classification metric is computed per class, then
    reported as both micro and macro averages.

    Args:
        predictions: Predicted probabilities, shape (n, n_classes).
        targets: Binary targets, shape (n, n_classes).
        threshold: Classification threshold.

    Returns:
        Dictionary with ``micro_*`` and ``macro_*`` prefixed metrics.
    """
    n_classes = predictions.shape[1]
    y_pred = (predictions >= threshold).astype(int)
    y_true = targets.astype(int)

    metrics: Dict[str, float] = {}

    # --- Micro-averaged metrics (pool all classes) ---
    y_true_flat = y_true.ravel()
    y_prob_flat = predictions.ravel()
    y_pred_flat = y_pred.ravel()

    metrics['micro_accuracy'] = float(
        accuracy_score(y_true_flat, y_pred_flat)
    )
    metrics['micro_f1'] = float(
        f1_score(y_true_flat, y_pred_flat, zero_division=0)
    )
    metrics['micro_precision'] = float(
        precision_score(y_true_flat, y_pred_flat, zero_division=0)
    )
    metrics['micro_recall_sensitivity'] = float(
        recall_score(y_true_flat, y_pred_flat, zero_division=0)
    )

    if len(np.unique(y_true_flat)) > 1:
        metrics['micro_auroc'] = float(
            roc_auc_score(y_true, predictions, average='micro')
        )
        metrics['micro_auprc'] = float(
            average_precision_score(y_true, predictions, average='micro')
        )
    else:
        metrics['micro_auroc'] = float('nan')
        metrics['micro_auprc'] = float('nan')

    cm_micro = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm_micro.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics['micro_specificity'] = float(spec)
    metrics['micro_false_positive_rate'] = float(1.0 - spec)
    metrics['micro_false_negative_rate'] = float(1.0 - sens)
    metrics['micro_ppv'] = float(
        tp / (tp + fp) if (tp + fp) > 0 else 0.0
    )
    metrics['micro_npv'] = float(
        tn / (tn + fn) if (tn + fn) > 0 else 0.0
    )
    metrics['micro_false_discovery_rate'] = float(
        fp / (fp + tp) if (fp + tp) > 0 else 0.0
    )
    metrics['micro_prevalence'] = float(np.mean(y_true_flat))

    # --- Macro-averaged metrics (per class then average) ---
    per_class: Dict[str, List[float]] = defaultdict(list)

    for c in range(n_classes):
        class_metrics = compute_binary_metrics(
            y_true[:, c], predictions[:, c], threshold
        )
        for key, val in class_metrics.items():
            per_class[key].append(val)

    for key, vals in per_class.items():
        arr = np.array(vals)
        # For AUROC/AUPRC, exclude NaN classes (single-class columns)
        if key in ('auroc', 'auprc'):
            valid = arr[~np.isnan(arr)]
            metrics[f'macro_{key}'] = (
                float(np.mean(valid)) if len(valid) > 0
                else float('nan')
            )
        else:
            metrics[f'macro_{key}'] = float(np.mean(arr))

    return metrics


# ------------------------------------------------------------------
# CSV discovery and parsing
# ------------------------------------------------------------------

def get_fold_names(experiment_dir: str) -> List[str]:
    """Discover fold directories under the experiment directory.

    Args:
        experiment_dir: Path to ``{model_dir}/{experiment_name}``.

    Returns:
        Sorted list of fold directory names matching ``fold\\d+``,
        excluding ``fold0`` (reserved for hyperparameter tuning).
    """
    folds = []
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if re.match(r'fold\d+', item) and os.path.isdir(item_path):
            if item != 'fold0':
                folds.append(item)
    folds.sort()
    return folds


def discover_splits(
    experiment_dir: str,
    fold_names: List[str]
) -> List[str]:
    """Discover which data splits have prediction CSVs.

    Args:
        experiment_dir: Path to ``{model_dir}/{experiment_name}``.
        fold_names: List of fold directory names.

    Returns:
        Sorted list of unique split names found across all folds
        and tasks (e.g. ``['test', 'train', 'val']``).
    """
    splits = set()
    for fold in fold_names:
        for task in TASKS:
            pattern = os.path.join(
                experiment_dir, fold, task, f'{task}_*_finetuned_output.csv'
            )
            for path in glob.glob(pattern):
                fname = os.path.basename(path)
                # Pattern: {task}_{split}_finetuned_output.csv
                match = re.match(
                    rf'^{re.escape(task)}_(.+)_finetuned_output\.csv$',
                    fname
                )
                if match:
                    splits.add(match.group(1))
    return sorted(splits)


def read_csv(path: str) -> Optional[np.ndarray]:
    """Read a prediction CSV and return its contents as a numpy array.

    Args:
        path: Path to the CSV file.

    Returns:
        Numpy array of shape (n_rows, n_cols), or ``None`` if the
        file does not exist.
    """
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    return df.values, list(df.columns)


def parse_predictions_targets(
    data: np.ndarray,
    columns: List[str],
    task: str
) -> tuple:
    """Split a CSV array into predictions and targets.

    Args:
        data: Full array from CSV, shape (n, n_cols).
        columns: Column names from the CSV header.
        task: One of ``'mortality'``, ``'length_of_stay'``,
            ``'phenotype'``.

    Returns:
        Tuple of (predictions, targets) numpy arrays.
    """
    pred_idx = [
        i for i, c in enumerate(columns)
        if c.startswith('pred') or c.startswith('prediction')
    ]
    targ_idx = [
        i for i, c in enumerate(columns)
        if c.startswith('target')
    ]
    return data[:, pred_idx], data[:, targ_idx]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute evaluation metrics from finetuned '
                    'TransEHR2 prediction CSVs'
    )
    parser.add_argument(
        'experiment_name', type=str,
        help='Name of the experiment (locates CSVs under model_dir)'
    )
    parser.add_argument(
        '--model_dir', type=str, default='./models',
        help='Root directory containing saved model weights and '
             'prediction CSVs (default: ./models)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold for binary metrics '
             '(default: 0.5)'
    )
    args = parser.parse_args()

    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        print(f"ERROR: Experiment directory not found: {experiment_dir}")
        exit(1)

    fold_names = get_fold_names(experiment_dir)
    if not fold_names:
        print(f"ERROR: No fold directories found in {experiment_dir}")
        exit(1)

    splits = discover_splits(experiment_dir, fold_names)
    if not splits:
        print(f"ERROR: No prediction CSVs found in {experiment_dir}")
        exit(1)

    print(f"Experiment: {args.experiment_name}")
    print(f"Folds: {fold_names}")
    print(f"Splits: {splits}")
    print(f"Threshold: {args.threshold}\n")

    for split in splits:
        print(f"{'=' * 60}")
        print(f"Split: {split}")
        print(f"{'=' * 60}")

        # Collect per-fold metrics for each task
        # Structure: {task: {metric_name: [fold1_val, fold2_val, ...]}}
        split_results: Dict[str, Dict[str, list]] = {}

        for task in TASKS:
            fold_metrics_list: List[Optional[Dict[str, float]]] = []
            task_has_data = False

            for fold in fold_names:
                csv_path = os.path.join(
                    experiment_dir, fold, task,
                    f'{task}_{split}_finetuned_output.csv'
                )
                result = read_csv(csv_path)
                if result is None:
                    fold_metrics_list.append(None)
                    continue

                data, columns = result
                predictions, targets = parse_predictions_targets(
                    data, columns, task
                )

                if task == 'mortality':
                    fold_metrics = compute_mortality_metrics(
                        predictions, targets, args.threshold
                    )
                elif task == 'length_of_stay':
                    fold_metrics = compute_los_metrics(
                        predictions, targets
                    )
                elif task == 'phenotype':
                    fold_metrics = compute_phenotype_metrics(
                        predictions, targets, args.threshold
                    )
                else:
                    fold_metrics = {}

                fold_metrics_list.append(fold_metrics)
                task_has_data = True

            if not task_has_data:
                continue

            # Collect metric values across folds into lists
            # Use the first non-None fold to get metric names
            metric_names = None
            for fm in fold_metrics_list:
                if fm is not None:
                    metric_names = list(fm.keys())
                    break
            if metric_names is None:
                continue

            task_results: Dict[str, list] = {}
            for name in metric_names:
                task_results[name] = []
                for fm in fold_metrics_list:
                    if fm is not None:
                        task_results[name].append(fm[name])
                    else:
                        task_results[name].append(float('nan'))

            split_results[task] = task_results

            # Print summary
            print(f"\n  {task}:")
            for name in metric_names:
                vals = np.array(task_results[name])
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    print(f"    {name}: "
                          f"{np.mean(valid):.4f} "
                          f"+/- {np.std(valid, ddof=1) / np.sqrt(len(valid)):.4f} "
                          f"(SEM, n={len(valid)})")
                else:
                    print(f"    {name}: no valid values")

        # Write YAML
        output = {
            'split': split,
            'experiment': args.experiment_name,
            'n_folds': len(fold_names),
            'folds': fold_names,
            'threshold': args.threshold
        }

        for task, task_metrics in split_results.items():
            # Convert numpy floats to native Python floats for YAML
            output[task] = {
                name: [round(v, 6) for v in vals]
                for name, vals in task_metrics.items()
            }

        yaml_path = os.path.join(
            experiment_dir, f'{split}_evaluation.yaml'
        )
        with open(yaml_path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        print(f"\n  -> {yaml_path}")

    print(f"\n{'=' * 60}")
    print("Done.")
