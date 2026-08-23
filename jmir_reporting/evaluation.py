"""Load per-fold predictions and compute evaluation metrics.

Reads the prediction CSVs written by ``dump_finetuned_predictions.py``
at ``{model_dir}/{experiment}/{fold}/{task}/{task}_{split}_finetuned_output.csv``
and computes the same metrics as ``evaluate_finetuned_predictions.py``,
with two additions needed for the manuscript tables:

* the decision threshold may be calibrated on a held-out split rather
  than fixed at 0.5, and
* for the multi-label phenotype task the threshold may be calibrated
  separately for each label.
"""

import glob
import math
import os
import re

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

TASKS = ('mortality', 'length_of_stay', 'phenotype')


# ------------------------------------------------------------------
# Locating experiments, folds and prediction CSVs
# ------------------------------------------------------------------

def resolve_experiment_dir(model_dir, number):
    """Find the directory for an experiment given its number.

    Args:
        model_dir: Directory holding one subdirectory per experiment.
        number: The experiment number, e.g. ``3`` for
            ``experiment3_nohistory``.

    Returns:
        Absolute path to the experiment directory.

    Raises:
        FileNotFoundError: If no directory matches.
        ValueError: If more than one directory matches.
    """
    pattern = os.path.join(model_dir, f'experiment{number}_*')
    matches = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    if not matches:
        raise FileNotFoundError(
            f'No experiment directory matching experiment{number}_* '
            f'under {model_dir}'
        )
    if len(matches) > 1:
        names = ', '.join(os.path.basename(m) for m in matches)
        raise ValueError(
            f'Experiment number {number} is ambiguous; matched: {names}'
        )
    return os.path.abspath(matches[0])


def discover_folds(experiment_dir):
    """List the cross-validation fold directories of an experiment.

    ``fold0`` is reserved for hyperparameter tuning and is excluded.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        Fold directory names sorted by their numeric suffix.
    """
    folds = []
    for item in os.listdir(experiment_dir):
        if not re.fullmatch(r'fold\d+', item):
            continue
        if item == 'fold0':
            continue
        if os.path.isdir(os.path.join(experiment_dir, item)):
            folds.append(item)
    folds.sort(key=lambda name: int(name[4:]))
    return folds


def prediction_csv_path(experiment_dir, fold, task, split):
    """Build the path of one prediction CSV.

    Args:
        experiment_dir: Path to the experiment directory.
        fold: Fold directory name, e.g. ``'fold1'``.
        task: One of ``'mortality'``, ``'length_of_stay'``,
            ``'phenotype'``.
        split: Data split name, e.g. ``'test'``.

    Returns:
        The path to the CSV, which may not exist.
    """
    return os.path.join(
        experiment_dir, fold, task,
        f'{task}_{split}_finetuned_output.csv'
    )


def load_predictions(experiment_dir, fold, task, split):
    """Load predictions and targets from one prediction CSV.

    Columns whose names begin with ``pred`` hold model outputs and
    columns whose names begin with ``target`` hold ground truth, in
    matching order.

    Args:
        experiment_dir: Path to the experiment directory.
        fold: Fold directory name.
        task: Task name.
        split: Data split name.

    Returns:
        A tuple ``(predictions, targets, label_names)`` where the arrays
        have shape ``(n_samples, n_labels)``.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the prediction and target columns do not pair up.
    """
    path = prediction_csv_path(experiment_dir, fold, task, split)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    pred_cols = [c for c in frame.columns if c.startswith('pred')]
    targ_cols = [c for c in frame.columns if c.startswith('target')]

    if not pred_cols or not targ_cols:
        raise ValueError(
            f'{path}: expected columns prefixed "pred" and "target", '
            f'found {list(frame.columns)}'
        )
    if len(pred_cols) != len(targ_cols):
        raise ValueError(
            f'{path}: {len(pred_cols)} prediction columns but '
            f'{len(targ_cols)} target columns'
        )

    predictions = frame[pred_cols].to_numpy(dtype=np.float64)
    targets = frame[targ_cols].to_numpy(dtype=np.float64)
    label_names = [re.sub(r'^pred(iction)?_?', '', c) or c
                   for c in pred_cols]
    return predictions, targets, label_names


# ------------------------------------------------------------------
# Threshold calibration
# ------------------------------------------------------------------

def prevalence_matched_threshold(probabilities, targets):
    """Find the threshold at which predicted and observed prevalence agree.

    The threshold is the ``1 - prevalence`` quantile of the predicted
    probabilities, so that the number of samples classified positive
    equals the number of observed positives. This makes the predicted
    positive count honest without optimising any performance metric, and
    it is stable to estimate even for rare labels because it inverts a
    quantile of a smooth distribution rather than maximising a jagged
    objective.

    Args:
        probabilities: Predicted probabilities, shape (n,).
        targets: Binary targets, shape (n,).

    Returns:
        The threshold. Falls back to 0.5 when the label has no positive
        or no negative examples.
    """
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    n = probs.size
    n_positive = int(round(float(np.sum(y))))

    if n == 0 or n_positive <= 0 or n_positive >= n:
        return 0.5

    # The n_positive-th largest probability is the smallest score that
    # still gets classified positive.
    ordered = np.sort(probs)[::-1]
    return float(ordered[n_positive - 1])


def calibrate_thresholds(probabilities, targets, mode, scope='per-label'):
    """Choose one decision threshold per label.

    Args:
        probabilities: Predicted probabilities, shape (n, n_labels).
        targets: Binary targets, shape (n, n_labels).
        mode: Either the string ``'prevalence'`` or a float giving a
            fixed threshold.
        scope: ``'per-label'`` calibrates each label independently;
            ``'global'`` pools all labels and returns a single threshold
            repeated across labels.

    Returns:
        An array of thresholds, shape (n_labels,).

    Raises:
        ValueError: If ``mode`` or ``scope`` is not recognised.
    """
    probs = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
    y = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    n_labels = probs.shape[1]

    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return np.full(n_labels, float(mode))

    if mode != 'prevalence':
        raise ValueError(
            f'Unknown threshold mode {mode!r}; expected "prevalence" or '
            f'a number'
        )

    if scope == 'global':
        shared = prevalence_matched_threshold(probs.ravel(), y.ravel())
        return np.full(n_labels, shared)

    if scope != 'per-label':
        raise ValueError(
            f'Unknown threshold scope {scope!r}; expected "per-label" or '
            f'"global"'
        )

    return np.array([
        prevalence_matched_threshold(probs[:, c], y[:, c])
        for c in range(n_labels)
    ])


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def concordance_index(predicted, observed):
    """Compute Harrell's concordance index for uncensored data.

    Pairs whose observed values are tied are excluded; pairs whose
    predicted values are tied count as half concordant. This is the same
    definition used by ``evaluate_finetuned_predictions.py`` but is
    computed from Kendall's tau-b rather than by enumerating pairs, so it
    runs in O(n log n) instead of O(n^2).

    Args:
        predicted: Predicted values, shape (n,).
        observed: Observed values, shape (n,).

    Returns:
        The concordance index in [0, 1], or NaN when no comparable pairs
        exist.
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    obs = np.asarray(observed, dtype=np.float64).ravel()

    n = pred.size
    if n < 2:
        return float('nan')

    total_pairs = n * (n - 1) / 2.0
    ties_obs = _tied_pairs(obs)
    ties_pred = _tied_pairs(pred)

    comparable = total_pairs - ties_obs
    if comparable <= 0:
        return float('nan')

    # tau_b = (C - D) / sqrt((n0 - ties_obs) * (n0 - ties_pred)), so the
    # excess of concordant over discordant pairs can be recovered from it.
    tau = scipy_stats.kendalltau(obs, pred, variant='b').statistic
    if math.isnan(tau):
        return float('nan')

    excess = tau * math.sqrt(
        (total_pairs - ties_obs) * (total_pairs - ties_pred)
    )
    return float(0.5 * (1.0 + excess / comparable))


def _tied_pairs(values):
    """Count the pairs of observations that share the same value.

    Args:
        values: A one-dimensional array.

    Returns:
        The number of tied pairs, as a float.
    """
    counts = np.unique(values, return_counts=True)[1].astype(np.float64)
    return float(np.sum(counts * (counts - 1.0) / 2.0))


def binary_metrics(y_true, y_prob, threshold):
    """Compute the full set of binary classification metrics.

    Args:
        y_true: Binary targets, shape (n,).
        y_prob: Predicted probabilities, shape (n,).
        threshold: The decision threshold.

    Returns:
        A dict mapping metric name to value. Metric names and formulas
        match ``evaluate_finetuned_predictions.py``.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (p >= threshold).astype(int)

    out = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall_sensitivity': float(
            recall_score(y, y_pred, zero_division=0)
        ),
    }

    if np.unique(y).size > 1:
        out['auroc'] = float(roc_auc_score(y, p))
        out['auprc'] = float(average_precision_score(y, p))
    else:
        out['auroc'] = float('nan')
        out['auprc'] = float('nan')

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    out['specificity'] = float(specificity)
    out['false_positive_rate'] = float(1.0 - specificity)
    out['false_negative_rate'] = float(1.0 - sensitivity)
    out['ppv'] = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    out['npv'] = float(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
    out['false_discovery_rate'] = float(
        fp / (fp + tp) if (fp + tp) > 0 else 0.0
    )
    out['prevalence'] = float(np.mean(y))
    return out


def mortality_metrics(predictions, targets, thresholds):
    """Compute in-hospital mortality metrics for one fold.

    Args:
        predictions: Predicted probabilities, shape (n, 1).
        targets: Binary targets, shape (n, 1).
        thresholds: Array of length 1 holding the decision threshold.

    Returns:
        A dict mapping metric name to value.
    """
    return binary_metrics(
        targets[:, 0], predictions[:, 0], float(thresholds[0])
    )


def length_of_stay_metrics(predictions, targets):
    """Compute length-of-stay regression metrics for one fold.

    Args:
        predictions: Predicted length of stay in hours, shape (n, 1).
        targets: Observed length of stay in hours, shape (n, 1).

    Returns:
        A dict with ``mean_absolute_error`` and ``concordance_index``.
    """
    pred = predictions[:, 0]
    obs = targets[:, 0]
    return {
        'mean_absolute_error': float(mean_absolute_error(obs, pred)),
        'concordance_index': concordance_index(pred, obs),
    }


def phenotype_metrics(predictions, targets, thresholds):
    """Compute multi-label phenotype metrics for one fold.

    Every binary metric is computed once over the pooled labels
    (micro average) and once per label and then averaged (macro
    average), matching ``evaluate_finetuned_predictions.py``.

    Args:
        predictions: Predicted probabilities, shape (n, n_labels).
        targets: Binary targets, shape (n, n_labels).
        thresholds: Per-label decision thresholds, shape (n_labels,).

    Returns:
        A dict with ``micro_``- and ``macro_``-prefixed metric names.
    """
    y_true = targets.astype(int)
    n_labels = predictions.shape[1]
    y_pred = (predictions >= thresholds[np.newaxis, :]).astype(int)

    out = {}

    flat_true = y_true.ravel()
    flat_pred = y_pred.ravel()
    flat_prob = predictions.ravel()

    out['micro_accuracy'] = float(accuracy_score(flat_true, flat_pred))
    out['micro_f1'] = float(f1_score(flat_true, flat_pred, zero_division=0))
    out['micro_precision'] = float(
        precision_score(flat_true, flat_pred, zero_division=0)
    )
    out['micro_recall_sensitivity'] = float(
        recall_score(flat_true, flat_pred, zero_division=0)
    )

    if np.unique(flat_true).size > 1:
        out['micro_auroc'] = float(
            roc_auc_score(y_true, predictions, average='micro')
        )
        out['micro_auprc'] = float(
            average_precision_score(y_true, predictions, average='micro')
        )
    else:
        out['micro_auroc'] = float('nan')
        out['micro_auprc'] = float('nan')

    tp = int(np.sum((flat_true == 1) & (flat_pred == 1)))
    tn = int(np.sum((flat_true == 0) & (flat_pred == 0)))
    fp = int(np.sum((flat_true == 0) & (flat_pred == 1)))
    fn = int(np.sum((flat_true == 1) & (flat_pred == 0)))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    out['micro_specificity'] = float(specificity)
    out['micro_false_positive_rate'] = float(1.0 - specificity)
    out['micro_false_negative_rate'] = float(1.0 - sensitivity)
    out['micro_ppv'] = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    out['micro_npv'] = float(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
    out['micro_false_discovery_rate'] = float(
        fp / (fp + tp) if (fp + tp) > 0 else 0.0
    )
    out['micro_prevalence'] = float(np.mean(flat_true))

    per_label = {}
    for c in range(n_labels):
        label_metrics = binary_metrics(
            y_true[:, c], predictions[:, c], float(thresholds[c])
        )
        for key, value in label_metrics.items():
            per_label.setdefault(key, []).append(value)

    for key, values in per_label.items():
        arr = np.asarray(values, dtype=np.float64)
        if key in ('auroc', 'auprc'):
            # Labels with a single observed class contribute no value.
            arr = arr[~np.isnan(arr)]
        out[f'macro_{key}'] = (
            float(np.mean(arr)) if arr.size else float('nan')
        )

    return out


# ------------------------------------------------------------------
# Per-experiment evaluation
# ------------------------------------------------------------------

class ExperimentResult:
    """Per-fold metric values for one experiment on one task.

    Attributes:
        number: The experiment number.
        name: The experiment directory name.
        folds: Fold directory names, in order.
        metrics: Dict mapping metric name to a list of per-fold values.
        thresholds: Dict mapping fold name to the array of thresholds
            used, or ``None`` for regression tasks.
        label_names: Label names read from the CSV header.
    """

    __slots__ = ('number', 'name', 'folds', 'metrics', 'thresholds',
                 'label_names')

    def __init__(self, number, name, folds, metrics, thresholds,
                 label_names):
        self.number = number
        self.name = name
        self.folds = folds
        self.metrics = metrics
        self.thresholds = thresholds
        self.label_names = label_names

    def values(self, metric):
        """Return the per-fold values of one metric as an array.

        Args:
            metric: The metric name.

        Returns:
            A numpy array of per-fold values.

        Raises:
            KeyError: If the metric was not computed.
        """
        return np.asarray(self.metrics[metric], dtype=np.float64)

    def threshold_summary(self, precision=3):
        """Summarise the thresholds actually used across folds.

        Args:
            precision: Decimal places for the reported values.

        Returns:
            A human-readable string, or ``None`` for regression tasks.
        """
        if self.thresholds is None:
            return None
        stacked = np.concatenate([
            np.atleast_1d(self.thresholds[f]) for f in self.folds
        ])
        if np.allclose(stacked, stacked[0]):
            return f'{stacked[0]:.{precision}f}'
        return (f'median {np.median(stacked):.{precision}f}, range '
                f'{stacked.min():.{precision}f}-'
                f'{stacked.max():.{precision}f}')


def evaluate_experiment(model_dir, number, task, split, threshold_mode,
                        calibration_split='val', threshold_scope='per-label',
                        folds=None):
    """Evaluate one experiment on one task across its folds.

    For classification tasks the decision threshold is calibrated on
    ``calibration_split`` within each fold and then applied to the
    reported ``split``, so no threshold is ever chosen on the split being
    reported.

    Args:
        model_dir: Directory holding one subdirectory per experiment.
        number: The experiment number.
        task: One of ``'mortality'``, ``'length_of_stay'``,
            ``'phenotype'``.
        split: The data split to report, usually ``'test'``.
        threshold_mode: ``'prevalence'`` or a fixed float. Ignored for
            ``'length_of_stay'``.
        calibration_split: Split used to calibrate the threshold when
            ``threshold_mode`` is ``'prevalence'``.
        threshold_scope: ``'per-label'`` or ``'global'``; only affects
            the multi-label phenotype task.
        folds: Explicit fold names, or ``None`` to auto-discover.

    Returns:
        An :class:`ExperimentResult`.

    Raises:
        FileNotFoundError: If the experiment, its folds, or a required
            prediction CSV is missing.
    """
    experiment_dir = resolve_experiment_dir(model_dir, number)
    name = os.path.basename(experiment_dir)
    fold_names = list(folds) if folds else discover_folds(experiment_dir)

    if not fold_names:
        raise FileNotFoundError(
            f'{name}: no fold directories found. The reporting scripts '
            f'read the per-fold prediction CSVs written by '
            f'dump_finetuned_predictions.py, not the aggregated '
            f'*_evaluation.yaml files.'
        )

    metrics = {}
    thresholds = None if task == 'length_of_stay' else {}
    label_names = None

    for fold in fold_names:
        predictions, targets, labels = load_predictions(
            experiment_dir, fold, task, split
        )
        if label_names is None:
            label_names = labels

        if task == 'length_of_stay':
            fold_metrics = length_of_stay_metrics(predictions, targets)
        else:
            fold_thresholds = _thresholds_for_fold(
                experiment_dir, fold, task, split, threshold_mode,
                calibration_split, threshold_scope,
                predictions, targets
            )
            thresholds[fold] = fold_thresholds
            if task == 'mortality':
                fold_metrics = mortality_metrics(
                    predictions, targets, fold_thresholds
                )
            else:
                fold_metrics = phenotype_metrics(
                    predictions, targets, fold_thresholds
                )

        for key, value in fold_metrics.items():
            metrics.setdefault(key, []).append(value)

    return ExperimentResult(
        number, name, fold_names, metrics, thresholds, label_names
    )


def _thresholds_for_fold(experiment_dir, fold, task, split, threshold_mode,
                         calibration_split, threshold_scope,
                         predictions, targets):
    """Determine the thresholds to apply to one fold of the report split.

    Args:
        experiment_dir: Path to the experiment directory.
        fold: Fold directory name.
        task: Task name.
        split: The split being reported.
        threshold_mode: ``'prevalence'`` or a fixed float.
        calibration_split: Split on which to calibrate.
        threshold_scope: ``'per-label'`` or ``'global'``.
        predictions: Predictions for the reported split, used only to
            establish the number of labels for fixed thresholds.
        targets: Targets for the reported split, unused when calibrating.

    Returns:
        An array of thresholds, one per label.

    Raises:
        FileNotFoundError: If the calibration split's CSV is missing.
    """
    scope = threshold_scope if task == 'phenotype' else 'global'

    if threshold_mode != 'prevalence':
        return calibrate_thresholds(
            predictions, targets, float(threshold_mode), scope
        )

    if calibration_split == split:
        raise ValueError(
            f'Refusing to calibrate the threshold on the split being '
            f'reported ({split!r}); choose a different '
            f'--calibration-split'
        )

    try:
        cal_predictions, cal_targets, _ = load_predictions(
            experiment_dir, fold, task, calibration_split
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f'Threshold calibration needs the {calibration_split!r} '
            f'split predictions for {os.path.basename(experiment_dir)}/'
            f'{fold}/{task}, but {exc} is missing. Pass a fixed '
            f'--threshold to skip calibration.'
        ) from None

    if cal_predictions.shape[1] != predictions.shape[1]:
        raise ValueError(
            f'{os.path.basename(experiment_dir)}/{fold}/{task}: the '
            f'{calibration_split!r} split has '
            f'{cal_predictions.shape[1]} labels but the {split!r} split '
            f'has {predictions.shape[1]}'
        )

    return calibrate_thresholds(
        cal_predictions, cal_targets, 'prevalence', scope
    )
