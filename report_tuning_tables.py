#!/usr/bin/env python3
"""Render the hyperparameter tuning supplementary tables from the runs on disk.

Reads each run's evaluation YAML, which records its own hyperparameters, so this works for a
sweep however its configs were produced -- a factorial spec, a one-at-a-time spec, or a config
list from `generate_finetune_grid.py` with no manifest at all. `report_tuning_results.py` reads
a manifest and ranks one hyperparameter at a time, which covers neither the factorial phase nor
the grid phase.

Two layouts, because the tables come in two shapes:

    grid    one cell per (row value, column value), one block per encoding arm. This is the
            learning-rate-by-half-life table: the pretraining sweep reported as validation
            loss, and the finetuning sweep reported as validation AUPRC.
    flat    one row per run, with every hyperparameter that varies as a column followed by the
            metrics. This is the table that reports a set of tuned configurations side by side
            rather than one hyperparameter's values.

A missing metric lists what the evaluation YAML does hold, since a metric name that does not
exist otherwise produces a table of blanks.

Usage:
    # Pretraining sweep, learning rate by half-life
    python report_tuning_tables.py 'phase2a_*' --layout grid --task pretrain \\
        --metric val:Optimization_Loss --row PRETRAIN_LEARNING_RATE \\
        --col PRETRAIN_LR_HALF_LIFE --table-number S4 \\
        --caption 'Pretraining validation set losses during hyperparameter tuning'

    # Finetuning sweep, learning rate by half-life
    python report_tuning_tables.py 'phase2b_*' --layout grid --task mortality \\
        --metric val:AUPRC --row FINETUNE_LEARNING_RATE --col FINETUNE_LR_HALF_LIFE \\
        --table-number S5

    # One row per tuned configuration
    python report_tuning_tables.py 'phase2c_*' --layout flat --task mortality \\
        --metrics val:AUROC,val:AUPRC --table-number S6
"""

import argparse
import fnmatch
import os
import sys

import yaml

from report_experiment_results import (BLOCKS, metric_value, read_run, render,
                                       resolve_model_dir, varying_hyperparameters)
from reporting.jmir.tables import Table, build_document, render_text


DEFAULT_TABLES_DIR = 'tables'

# The hyperparameter that names the encoding arm, which the grid layout blocks on.
ARM_KEY = 'POSITION_ENCODING'

# A run carrying one of these is a control, not a point in the rate-by-half-life plane: it sits
# at the reference cell's rate and schedule but replaces or freezes the encoder. Left in, it
# would be averaged into that cell alongside the grid run that belongs there. The flat layout
# keeps them, since there the controls are rows of their own.
CONTROL_MARKERS = ('FINETUNE_ENCODER_INIT', 'FINETUNE_FREEZE_ENCODER')

# Printed for a cell whose run produced no result, so a hole is visible rather than looking
# like a value of zero.
MISSING = '--'

# House wording for the things a tuning table names, so its output can be pasted rather than
# retyped. A key absent here falls back to its own name, title-cased.
HEADINGS = {
    'POSITION_ENCODING': 'Temporal Encoding Method',
    'PRETRAIN_LEARNING_RATE': 'Pretraining Learn Rate',
    'PRETRAIN_LR_HALF_LIFE': 'Pretraining Learn Rate Half-Life (Decay)',
    'FINETUNE_LEARNING_RATE': 'Finetuning Learn Rate',
    'FINETUNE_LR_HALF_LIFE': 'Finetuning Learn Rate Half-Life (Decay)',
    'CMPNT_MASK_RATIO': 'Embedding Component Mask Rate',
    'RECORD_MASK_RATIO': 'Record Mask Ratio',
    'THP_PRED_LOSS_TIME_WT': 'Transformer Hawkes Process Time Loss Weight',
}

METRIC_HEADINGS = {
    'val:AUPRC': 'Validation AUPRC',
    'val:AUROC': 'Validation AUROC',
    'test:AUPRC': 'Test AUPRC',
    'test:AUROC': 'Test AUROC',
    'val:Optimization_Loss': 'Validation loss',
    'val:Best_Epoch': 'Best epoch',
}

ARM_HEADINGS = {
    'additive': 'Temporal Positional Encoding (TPE)',
    'rope': 'Rotary Position Embedding (RoPE)',
}


def heading(key):
    """House wording for a hyperparameter, falling back to the key itself."""
    return HEADINGS.get(key, key.replace('_', ' ').title())


def metric_heading(spec):
    """House wording for a metric spec, falling back to the spec itself."""
    return METRIC_HEADINGS.get(spec, spec)


def discover(model_dir, patterns, fold, task):
    """Every run matching a pattern that has an evaluation on disk.

    Args:
        model_dir: The model tree.
        patterns: Experiment name patterns.
        fold: Fold name.
        task: Task name, or 'pretrain'.

    Returns:
        List of (name, parsed YAML), sorted by name.

    Raises:
        SystemExit: If no run matched, which otherwise renders an empty table.
    """
    if not os.path.isdir(model_dir):
        raise SystemExit(f'{model_dir} is not a directory. Check --model_dir.')
    names = sorted(entry for entry in os.listdir(model_dir)
                   if any(fnmatch.fnmatch(entry, pattern) for pattern in patterns))
    runs = [(name, read_run(model_dir, name, fold, task)) for name in names]
    found = [(name, data) for name, data in runs if data is not None]
    if not found:
        missing = len(runs)
        raise SystemExit(
            f'No evaluation found for any of the {missing} run(s) matching '
            f'{" ".join(patterns)} under {model_dir} at fold {fold}, task {task!r}. Check '
            f'--fold and --task; a pretraining sweep needs --task pretrain.'
        )
    return found


def available_metrics(data):
    """The block:metric specs one evaluation YAML actually carries."""
    specs = []
    for prefix, candidates in BLOCKS.items():
        for candidate in candidates:
            block = data.get(candidate)
            if isinstance(block, dict):
                specs.extend(f'{prefix}:{key}' for key in block)
    return sorted(set(specs))


def check_metrics(runs, specs):
    """Stop if a metric is absent everywhere, naming what is present instead.

    Args:
        runs: Sequence of (name, parsed YAML).
        specs: Requested block:metric specs.

    Raises:
        SystemExit: If a spec resolves nowhere.
    """
    for spec in specs:
        if any(metric_value(data, spec) is not None for _, data in runs):
            continue
        _, sample = runs[0]
        raise SystemExit(
            f'{spec!r} is not in any of these evaluations. Available: '
            f'{", ".join(available_metrics(sample)) or "nothing"}.'
        )


def hyperparameter(data, key):
    """One recorded hyperparameter of a run, or None."""
    return (data.get('hyperparameters') or {}).get(key)


def axis_values(runs, key):
    """The distinct values of one hyperparameter, ordered for an axis.

    Numerically where they are numbers, so a rate axis reads in order. None sorts last, being
    the flat schedule -- the limit of the decay axis rather than a missing value.
    """
    values = {hyperparameter(data, key) for _, data in runs}
    numeric = sorted(v for v in values if isinstance(v, (int, float)))
    other = sorted((str(v) for v in values if v is not None
                    and not isinstance(v, (int, float))))
    return numeric + other + ([None] if None in values else [])


def format_axis(value, key):
    """Axis label for one hyperparameter value.

    A half-life is in epochs and the published tables say so in the column heading, which is
    also what distinguishes it from a rate at a glance.
    """
    if value is None:
        return 'No decay'
    label = f'{value:g}' if isinstance(value, (int, float)) else str(value)
    if key.endswith('HALF_LIFE') and value is not None:
        return f'{label} Epochs'
    return label


def build_grid(runs, row_key, col_key, spec, number, caption, precision):
    """One block per encoding arm, rows by `row_key` and columns by `col_key`.

    Args:
        runs: Sequence of (name, parsed YAML).
        row_key: Hyperparameter on the rows.
        col_key: Hyperparameter on the columns.
        spec: The block:metric to put in each cell.
        number: Table number for the caption.
        caption: Caption text.
        precision: Decimal places for a cell.

    Returns:
        The assembled `Table`.
    """
    controls = [name for name, data in runs
                if any(hyperparameter(data, key) is not None for key in CONTROL_MARKERS)]
    if controls:
        print(f'  excluding {len(controls)} control run(s) from the grid: '
              f'{", ".join(controls)}')
        runs = [(name, data) for name, data in runs if name not in set(controls)]

    rows = axis_values(runs, row_key)
    cols = axis_values(runs, col_key)
    arms = axis_values(runs, ARM_KEY) or [None]

    table = Table(number, caption, heading(row_key),
                  [format_axis(value, col_key) for value in cols])
    table.add_footnote(f'Cells are {metric_heading(spec)}. {MISSING} marks a run with no '
                       f'result on disk.')

    for arm in arms:
        if len(arms) > 1 or arm is not None:
            table.add_category(ARM_HEADINGS.get(arm, str(arm)))
        for row_value in rows:
            cells = []
            for col_value in cols:
                matches = [data for _, data in runs
                           if hyperparameter(data, ARM_KEY) == arm
                           and hyperparameter(data, row_key) == row_value
                           and hyperparameter(data, col_key) == col_value]
                values = [metric_value(data, spec) for data in matches]
                values = [value for value in values if value is not None]
                if not values:
                    cells.append(MISSING)
                else:
                    # More than one run per cell means seed repeats; report the mean and say so
                    # rather than silently picking one.
                    mean = sum(values) / len(values)
                    cells.append(f'{mean:.{precision}f}'
                                 + (f' (n={len(values)})' if len(values) > 1 else ''))
            table.add_row(format_axis(row_value, row_key), cells,
                          level=1 if len(arms) > 1 else 0)
    return table


def build_flat(runs, specs, number, caption, precision):
    """One row per run: every hyperparameter that varies, then the metrics.

    Args:
        runs: Sequence of (name, parsed YAML).
        specs: The block:metric specs to report.
        number: Table number for the caption.
        caption: Caption text.
        precision: Decimal places for a metric.

    Returns:
        The assembled `Table`.
    """
    keys = varying_hyperparameters(runs)
    columns = [heading(key) for key in keys] + [metric_heading(s) for s in specs]
    table = Table(number, caption, 'Run', columns)

    for name, data in runs:
        cells = []
        for key in keys:
            value = hyperparameter(data, key)
            if value is None:
                cells.append('No decay' if key.endswith('HALF_LIFE') else MISSING)
            elif key == ARM_KEY:
                cells.append(ARM_HEADINGS.get(value, render(value)))
            else:
                cells.append(format_axis(value, key))
        for spec in specs:
            value = metric_value(data, spec)
            cells.append(MISSING if value is None else f'{value:.{precision}f}')
        table.add_row(name, cells)
    return table


def write_csv(path, table):
    """Write the same numbers as a CSV, one line per row."""
    import csv as csv_module
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv_module.writer(handle)
        writer.writerow([table.stub_head] + table.columns)
        group = ''
        for row in table.rows:
            if row.kind == 'category':
                group = row.label
                continue
            writer.writerow(([group] if group else []) + [row.label] + row.cells)
    print(f'Wrote {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Render the hyperparameter tuning supplementary tables'
    )
    parser.add_argument('patterns', nargs='+',
                        help="Experiment name patterns, e.g. 'phase2a_*'. Quote them so the "
                             'shell does not expand them against the working directory.')
    parser.add_argument('--layout', choices=('grid', 'flat'), default='grid',
                        help='grid pivots two hyperparameters, one block per encoding arm; '
                             'flat gives one row per run (default: grid)')
    parser.add_argument('--model_dir', default=None,
                        help='Model tree. Defaults to MODEL_DIR in --base_config.')
    parser.add_argument('--base_config',
                        default='TransEHR2/configs/experiments/tuning/phase2_base.yaml',
                        help='Config to read MODEL_DIR from')
    parser.add_argument('--fold', default='fold0', help='Fold to read (default: fold0)')
    parser.add_argument('--task', default='mortality',
                        help="Task name, or 'pretrain' for the pretraining evaluation")
    parser.add_argument('--metric', default=None,
                        help='grid layout: the single block:metric each cell holds')
    parser.add_argument('--metrics', default=None,
                        help='flat layout: comma-separated block:metric specs')
    parser.add_argument('--row', default='PRETRAIN_LEARNING_RATE',
                        help='grid layout: hyperparameter on the rows')
    parser.add_argument('--col', default='PRETRAIN_LR_HALF_LIFE',
                        help='grid layout: hyperparameter on the columns')
    parser.add_argument('--precision', type=int, default=4,
                        help='Decimal places for a reported value (default: 4)')
    parser.add_argument('--table-number', default='S', help='Table number for the caption')
    parser.add_argument('--caption', default='Hyperparameter tuning results',
                        help='Caption text, without the "Table N." prefix')
    parser.add_argument('--output', default=None,
                        help=f'Word output path. Defaults to {DEFAULT_TABLES_DIR}/'
                             f'<pattern>_tuning.docx')
    parser.add_argument('--csv', default=None,
                        help=f'CSV output path. Defaults to {DEFAULT_TABLES_DIR}/'
                             f'<pattern>_tuning.csv')
    parser.add_argument('--caption-style', default=None,
                        help='Paragraph style for the caption in the Word document')
    parser.add_argument('--no_files', action='store_true',
                        help='Print the table without writing anything')
    args = parser.parse_args(argv)

    model_dir = resolve_model_dir(args.model_dir, args.base_config)
    runs = discover(model_dir, args.patterns, args.fold, args.task)
    print(f'{len(runs)} run(s) under {model_dir}, fold {args.fold}, task {args.task!r}')

    if args.layout == 'grid':
        spec = args.metric or ('val:Optimization_Loss' if args.task == 'pretrain'
                               else 'val:AUPRC')
        check_metrics(runs, [spec])
        table = build_grid(runs, args.row, args.col, spec, args.table_number, args.caption,
                           args.precision)
    else:
        specs = [item.strip() for item in
                 (args.metrics or 'val:AUROC,val:AUPRC').split(',') if item.strip()]
        check_metrics(runs, specs)
        table = build_flat(runs, specs, args.table_number, args.caption, args.precision)

    render_text(table)

    if args.no_files:
        return 0

    stem = args.patterns[0].replace('*', '').rstrip('_') or 'tuning'
    csv_path = args.csv or os.path.join(DEFAULT_TABLES_DIR, f'{stem}_tuning.csv')
    docx_path = args.output or os.path.join(DEFAULT_TABLES_DIR, f'{stem}_tuning.docx')
    write_csv(csv_path, table)
    os.makedirs(os.path.dirname(docx_path) or '.', exist_ok=True)
    build_document([table], docx_path, caption_style=args.caption_style)
    print(f'Wrote {docx_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
