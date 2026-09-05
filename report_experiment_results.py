"""Tabulate finished experiment results from their evaluation YAMLs.

Reads ``<MODEL_DIR>/<experiment>/<fold>/<task>/evaluation/evaluation_<task>.yaml`` for every
experiment whose name matches a pattern, and prints one row per run. Any recorded
hyperparameter that differs across the selected runs becomes a column, so a grid describes
itself.

    python report_experiment_results.py 'phase2b_*'
    python report_experiment_results.py 'phase2h_*' --paired
    python report_experiment_results.py 'phase2b_additive_*' --csv wave_b_additive.csv

``--paired`` pairs runs by the seed in their names and reports the mean difference between
two groups with a confidence interval. Pairing removes the run-to-run variance the groups
share.
"""

import argparse
import csv
import fnmatch
import os
import re
import statistics
import sys

import yaml
from scipy import stats

from hp_tuning.reporting import print_table

DEFAULT_BASE_CONFIG = 'TransEHR2/configs/experiments/tuning/phase2_base.yaml'
DEFAULT_METRICS = 'val:Best_Epoch,val:AUPRC,val:AUROC,test:AUPRC,test:AUROC,test:F1_Score'
# A finetuned task writes *_scores; pretraining writes *_losses. One prefix reads either, so
# a metric spec does not have to know which stage produced the file.
BLOCKS = {
    'train': ('train_scores', 'train_losses'),
    'val': ('validation_scores', 'val_losses'),
    'test': ('test_scores',),
}
PRETRAIN_METRICS = 'val:Best_Epoch,val:Optimization_Loss,val:Generator_Loss,val:THP_NLL_Loss'
SEED_PATTERN = re.compile(r'^(?P<group>.+)_seed(?P<seed>\d+)$')


def resolve_model_dir(given, base_config):
    """Find the tree the runs wrote into.

    Args:
        given: An explicit path, or None.
        base_config: Config to read MODEL_DIR from when no path is given.

    Returns:
        str: The model directory.

    Raises:
        SystemExit: If neither source yields one.
    """
    if given:
        return given
    if not os.path.exists(base_config):
        sys.exit(f"No --model_dir given and {base_config} does not exist to read it from.")
    with open(base_config) as f_in:
        model_dir = yaml.safe_load(f_in).get('MODEL_DIR')
    if not model_dir:
        sys.exit(f"{base_config} does not define MODEL_DIR.")
    return model_dir


def read_run(model_dir, name, fold, task):
    """Read one run's evaluation YAML.

    Args:
        model_dir: The model tree.
        name: EXPERIMENT_NAME.
        fold: Fold name.
        task: Task name, or 'pretrain' for the pretraining evaluation.

    Returns:
        dict: The parsed YAML, or None if it is not there.
    """
    if task == 'pretrain':
        path = os.path.join(model_dir, name, fold, 'pretrained', 'evaluation',
                            'evaluation_pretrained.yaml')
    else:
        path = os.path.join(model_dir, name, fold, task, 'evaluation',
                            f'evaluation_{task}.yaml')
    if not os.path.exists(path):
        return None
    with open(path) as f_in:
        return yaml.safe_load(f_in)


def metric_value(data, spec):
    """Pull one ``block:metric`` value out of a parsed evaluation YAML.

    Args:
        data: The parsed YAML.
        spec: A metric spec, e.g. 'val:AUPRC'.

    Returns:
        The value, or None if the block or the metric is absent.
    """
    block_name, _, metric = spec.partition(':')
    for candidate in BLOCKS.get(block_name, (block_name,)):
        block = data.get(candidate)
        if isinstance(block, dict) and metric in block:
            return block[metric]
    return None


def varying_hyperparameters(runs):
    """Recorded hyperparameters whose value is not the same in every run.

    Args:
        runs: Sequence of (name, parsed YAML) pairs.

    Returns:
        list: Hyperparameter names, in the order they appear.
    """
    seen = {}
    for _, data in runs:
        for key, value in (data.get('hyperparameters') or {}).items():
            seen.setdefault(key, set()).add(repr(value))
    return [key for key, values in seen.items() if len(values) > 1]


def render(value):
    """Format a cell.

    Args:
        value: Any cell value.

    Returns:
        str: Four decimal places for a fraction, plain otherwise.
    """
    if value is None:
        return '-'
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f'{int(value)}'
    # Fixed point down to a thousandth, which covers every metric here; below that the value
    # is a learning rate or a weight and fixed point would render it as zero.
    if abs(value) < 1e-3 or abs(value) >= 1000:
        return f'{value:g}'
    return f'{value:.4f}'


def paired_report(runs, spec):
    """Pair runs by the seed in their names and report the difference between two groups.

    Args:
        runs: Sequence of (name, parsed YAML) pairs.
        spec: The metric to compare, e.g. 'val:AUPRC'.

    Returns:
        int: 0 on success, 1 if the runs are not two seed-matched groups.
    """
    grouped = {}
    for name, data in runs:
        match = SEED_PATTERN.match(name)
        if match is None:
            print(f"  {name} has no _seed<N> suffix, so it cannot be paired.")
            continue
        grouped.setdefault(match['group'], {})[int(match['seed'])] = metric_value(data, spec)

    if len(grouped) != 2:
        print(f"\nPairing needs exactly two groups; found {len(grouped)}: {sorted(grouped)}")
        return 1

    (name_a, a), (name_b, b) = sorted(grouped.items())
    seeds = sorted(set(a) & set(b))
    unmatched = sorted(set(a) ^ set(b))
    if unmatched:
        print(f"\nSeeds present in only one group, excluded: {unmatched}")

    rows, differences = [], []
    for seed in seeds:
        if a[seed] is None or b[seed] is None:
            rows.append([str(seed), render(a[seed]), render(b[seed]), '-'])
            continue
        difference = a[seed] - b[seed]
        differences.append(difference)
        rows.append([str(seed), render(a[seed]), render(b[seed]), f'{difference:+.4f}'])

    print(f"\nPaired on {spec}, {name_a} minus {name_b}")
    print_table(rows, ['seed', name_a, name_b, 'difference'])

    if len(differences) < 2:
        print(f"\n  {len(differences)} usable pair(s): not enough to estimate a spread.")
        return 0

    n = len(differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    standard_error = sd / n ** 0.5
    half_width = stats.t.ppf(0.975, n - 1) * standard_error
    t_statistic = mean / standard_error if standard_error else float('inf')
    p_value = 2 * stats.t.sf(abs(t_statistic), n - 1)

    print(f"\n  mean difference   {mean:+.4f}")
    print(f"  95% CI            [{mean - half_width:+.4f}, {mean + half_width:+.4f}]")
    print(f"  sd of differences {sd:.4f}   n = {n}")
    print(f"  paired t          t({n - 1}) = {t_statistic:.2f}, p = {p_value:.3f}")
    if (mean - half_width) * (mean + half_width) <= 0:
        print("\n  The interval spans zero, so these two are not separated by this many "
              "repeats.")
    return 0


def main(argv=None):
    """Tabulate the matching runs.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('patterns', nargs='+',
                        help="Experiment name patterns, e.g. 'phase2b_*'. Quote them so the "
                             "shell does not expand them against the working directory.")
    parser.add_argument('--model_dir', default=None,
                        help=f'Model tree. Defaults to MODEL_DIR in {DEFAULT_BASE_CONFIG}')
    parser.add_argument('--base_config', default=DEFAULT_BASE_CONFIG,
                        help='Config to read MODEL_DIR from')
    parser.add_argument('--fold', default='fold0', help='Fold to read')
    parser.add_argument('--task', default='mortality',
                        help="Task to read, or 'pretrain' for the pretraining evaluation")
    parser.add_argument('--metrics', default=None,
                        help='Comma-separated block:metric specs. Defaults to the task\'s '
                             'ranking metrics, or to the pretraining losses under '
                             "--task pretrain.")
    parser.add_argument('--sort', default=None,
                        help='Metric spec to sort by, best first')
    parser.add_argument('--show', default=None,
                        help='Comma-separated hyperparameters to column even when every run '
                             'agrees on them. Hyperparameters that differ are always shown; '
                             'this is for confirming that the ones meant to be constant are.')
    parser.add_argument('--paired', action='store_true',
                        help='Pair runs by the seed in their names and compare two groups')
    parser.add_argument('--paired_metric', default='val:AUPRC',
                        help='Metric the pairing compares')
    parser.add_argument('--csv', default=None, help='Also write the table to this path')
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir, args.base_config)
    if not os.path.isdir(model_dir):
        sys.exit(f"Model directory does not exist: {model_dir}")

    names = sorted(
        name for name in os.listdir(model_dir)
        if any(fnmatch.fnmatch(name, pattern) for pattern in args.patterns)
    )
    if not names:
        sys.exit(f"No experiment under {model_dir} matches {args.patterns}")

    runs, missing = [], []
    for name in names:
        data = read_run(model_dir, name, args.fold, args.task)
        if data is None:
            missing.append(name)
        else:
            runs.append((name, data))

    print(f"Model dir:  {model_dir}")
    print(f"Fold:       {args.fold}    Task: {args.task}")
    print(f"Runs:       {len(runs)} of {len(names)} matched experiments have results\n")

    if not runs:
        print("Nothing to tabulate. The matched experiments have not written results yet.")
        return 1

    chosen = args.metrics or (PRETRAIN_METRICS if args.task == 'pretrain' else DEFAULT_METRICS)
    metrics = [spec.strip() for spec in chosen.split(',') if spec.strip()]
    hyperparameters = varying_hyperparameters(runs)
    for key in (spec.strip() for spec in (args.show or '').split(',') if spec.strip()):
        if key not in hyperparameters:
            hyperparameters.append(key)

    # A shared prefix says the same thing on every row, so it moves into the header.
    prefix = os.path.commonprefix([name for name, _ in runs])
    prefix = prefix if len(prefix) > 4 and len(runs) > 1 else ''
    if prefix:
        print(f"Names are shown without their common prefix {prefix!r}\n")

    if args.sort:
        # A loss is better when small and a score when large, so the direction follows the
        # metric name rather than being another thing to remember.
        descending = 'loss' not in args.sort.lower()
        runs.sort(key=lambda pair: (
            metric_value(pair[1], args.sort) is None,
            -(metric_value(pair[1], args.sort) or 0) if descending
            else (metric_value(pair[1], args.sort) or 0)))

    headers = (['experiment'] + [key.replace('FINETUNE_', 'FT_').replace('PRETRAIN_', 'PT_')
                                 for key in hyperparameters] + metrics)
    rows = [
        [name[len(prefix):] or name]
        + [render((data.get('hyperparameters') or {}).get(key)) for key in hyperparameters]
        + [render(metric_value(data, spec)) for spec in metrics]
        for name, data in runs
    ]
    print_table(rows, headers)

    if missing:
        print(f"\nNo results yet: {', '.join(missing)}")

    if args.csv:
        with open(args.csv, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")

    if args.paired:
        return paired_report(runs, args.paired_metric)
    return 0


if __name__ == '__main__':
    sys.exit(main())
