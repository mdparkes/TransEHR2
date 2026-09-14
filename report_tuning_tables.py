#!/usr/bin/env python3
"""Render the hyperparameter tuning supplementary tables from the runs on disk.

Reads each run's evaluation YAML, which records its own hyperparameters, so this works for a
sweep however its configs were produced -- a factorial spec, a one-at-a-time spec, or a config
list from `generate_finetune_grid.py` with no manifest at all. `report_tuning_results.py` reads
a manifest and ranks one hyperparameter at a time, which covers neither the factorial phase nor
the grid phase.

What a run records is `RECORDED_HYPERPARAMETERS` in run_experiment.py, a fixed list. A sweep
over anything outside it produces evaluations that do not say which cell they came from, and
every run then collapses into a single cell of the grid. `NAME_TOKENS` closes that for the
hyperparameters `generate_finetune_grid.py` writes into the cell name, so a table can still be
built from runs finished before the list was extended.

Two layouts, because the tables come in two shapes:

    grid    one cell per (row value, column value), one block per encoding arm. This is the
            learning-rate-by-half-life table: the pretraining sweep reported as validation
            loss, and the finetuning sweep reported as validation AUPRC.
    flat    one row per run: the encoding arm, the configuration it ran at, then the metrics.
            This is the table that reports a set of tuned configurations side by side rather
            than one hyperparameter's values. Every configuration column is reported whether
            or not it varies, since the settings a phase inherited are what make its rows
            comparable to another phase's, and the best row of each arm is emphasised.

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

    # One row per tuned configuration. --spec gives the order of the blocks and of the rows
    # within them, which the trial names do not carry.
    python report_tuning_tables.py 'phase2c_*' --layout flat --task mortality \\
        --metrics val:AUROC,val:AUPRC --table-number S6 \\
        --spec 'TransEHR2/configs/experiments/tuning/phase2c_*_spec.yaml' 
"""

import argparse
import fnmatch
import glob
import math
import os
import re
import sys

import yaml

from report_experiment_results import (BLOCKS, metric_value, read_run, render,
                                       resolve_model_dir, varying_hyperparameters)
from reporting.jmir.tables import Table, build_document, render_text, strip_markup


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
    'rope': 'RoPE',
}

# The flat layout carries the arm as its first column, where the tables abbreviate it and
# expand the abbreviation in a footnote.
ARM_SHORT = {
    'additive': 'TPE',
    'rope': 'RoPE',
}

ARM_FOOTNOTE = ('TPE \u2013 Temporal Positional Encoding; RoPE \u2013 Rotary Position '
                'Embedding')

# Footnotes a column heading carries, keyed by hyperparameter.
COLUMN_FOOTNOTES = {
    'RECORD_MASK_RATIO': 'Ratio of unobserved to observed records selected for masking '
                         'during self-supervised pretraining',
}

# The configuration columns of the flat layout, in order: the two schedules the phase carries
# in from the phases before it, then the hyperparameters it sweeps. Unlike the grid, a column
# is kept even when every run shares its value, because the table reports the configuration
# each row was run at rather than only what varied. A column no run carries a value for is
# dropped.
FLAT_COLUMNS = (
    'PRETRAIN_LEARNING_RATE',
    'PRETRAIN_LR_HALF_LIFE',
    'FINETUNE_LEARNING_RATE',
    'FINETUNE_LR_HALF_LIFE',
    'CMPNT_MASK_RATIO',
    'RECORD_MASK_RATIO',
    'THP_PRED_LOSS_TIME_WT',
)

# Where the trial configs are, for a hyperparameter neither the evaluation nor the cell name
# carries. A one-at-a-time sweep names only the hyperparameter it varies, so the settings it
# inherited are recoverable from nothing else.
DEFAULT_CONFIG_GLOB = os.path.join('TransEHR2', 'configs', 'experiments', '**', '*.yaml')

# The published tables head the rate axis with the bare quantity, not the stage: the stage is
# already in the caption, and each table reports one stage.
GRID_STUB = {
    'PRETRAIN_LEARNING_RATE': 'Learning Rate',
    'FINETUNE_LEARNING_RATE': 'Learning Rate',
    'PRETRAIN_LR_HALF_LIFE': 'Learning rate half-life (exponential decay)',
    'FINETUNE_LR_HALF_LIFE': 'Learning rate half-life (exponential decay)',
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


# The token `generate_finetune_grid.token` writes a value into the cell name as, for the
# hyperparameters a run's evaluation may not record. Tokens hold no underscore, so each runs to
# the next separator.
NAME_TOKENS = {
    'PRETRAIN_LEARNING_RATE': 'lr',
    'PRETRAIN_LR_HALF_LIFE': 'hl',
    'FINETUNE_LEARNING_RATE': 'lr',
    'FINETUNE_LR_HALF_LIFE': 'hl',
}


def decode_token(text):
    """The value a name token encodes, or None for a flat schedule.

    `token` replaces '.' with 'p' and '-' with 'm' to keep the name filename-safe, so 5e-05
    becomes '5em05'.

    Args:
        text: The token, without its prefix.

    Returns:
        The value as a float, or None if the token names a flat schedule.

    Raises:
        ValueError: If the token is neither, which keeps an unrelated name that happens to
            contain the prefix from being read as a value.
    """
    if text.lower() in ('flat', 'none'):
        return None
    return float(text.replace('p', '.').replace('m', '-'))


def from_name(name, key):
    """One hyperparameter's value read out of an experiment name.

    Args:
        name: The EXPERIMENT_NAME of the run.
        key: The hyperparameter to recover.

    Returns:
        The value, or None if the name carries no token for this key.
    """
    prefix = NAME_TOKENS.get(key)
    if prefix is None:
        return None
    match = re.search(rf'_{prefix}([^_]+)', name)
    if match is None:
        return None
    try:
        return decode_token(match.group(1))
    except ValueError:
        return None


def config_index(pattern):
    """Map experiment name to trial config, for a hyperparameter no run recorded.

    A name carries only what the sweep varied, so a one-at-a-time phase cannot be read back
    from names alone: the settings it inherited appear in no name and, if they are outside
    `RECORDED_HYPERPARAMETERS`, in no evaluation either. The config that produced the run
    holds all of them.

    Args:
        pattern: Recursive glob for the trial configs.

    Returns:
        Dict of experiment name to parsed config. A config unreadable or without a name of
        its own is skipped rather than failing the report.
    """
    index = {}
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path) as handle:
                config = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(config, dict):
            continue
        name = config.get('EXPERIMENT_NAME') or os.path.splitext(os.path.basename(path))[0]
        index.setdefault(name, config)
    return index


def hyperparameter(data, key, name=None, configs=None):
    """One hyperparameter of a run: its evaluation, else its config, else its name.

    Args:
        data: The run's parsed evaluation YAML.
        key: The hyperparameter to read.
        name: The run's EXPERIMENT_NAME, needed by both fallbacks. A key recorded as None is
            a flat schedule, not a missing value, so the fallbacks do not fire for it.
        configs: The dict `config_index` returns, or None to skip that fallback.

    Returns:
        The value, or None.
    """
    recorded = data.get('hyperparameters') or {}
    if key in recorded:
        return recorded[key]
    if name is None:
        return None
    config = (configs or {}).get(name) or {}
    if key in config:
        return config[key]
    return from_name(name, key)


def axis_values(runs, key, configs=None):
    """The distinct values of one hyperparameter, ordered for an axis.

    Numerically where they are numbers, so a rate axis reads in order. None sorts last, being
    the flat schedule -- the limit of the decay axis rather than a missing value.
    """
    values = {hyperparameter(data, key, name, configs) for name, data in runs}
    numeric = sorted(v for v in values if isinstance(v, (int, float)))
    other = sorted((str(v) for v in values if v is not None
                    and not isinstance(v, (int, float))))
    return numeric + other + ([None] if None in values else [])


def format_rate(value):
    """A learning rate in the published form, e.g. 0.0006 as 6x10<sup>-4</sup>.

    The mantissa is rounded before the trailing zeros are stripped, because dividing by a power
    of ten leaves values like 5.999999999999999 that would otherwise print in full.
    """
    if not isinstance(value, (int, float)) or value == 0:
        return str(value)
    exponent = math.floor(math.log10(abs(value)))
    mantissa = round(value / (10.0 ** exponent), 3)
    # Rounding can carry the mantissa to 10, which belongs in the exponent.
    if abs(mantissa) >= 10:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    text = f'{mantissa:g}'
    return f'{text}x10<sup>{exponent}</sup>'


def format_axis(value, key):
    """Axis label for one hyperparameter value.

    A half-life is in epochs and the published tables say so in the column heading, which is
    also what distinguishes it from a rate at a glance. A rate is written as a mantissa and a
    power of ten, which is the house form.
    """
    if value is None:
        return 'No Decay'
    if key.endswith('LEARNING_RATE'):
        return format_rate(value)
    if key.endswith('MASK_RATIO') and isinstance(value, (int, float)):
        # The tables set the masking ratios to a common width, so 0.5 reads as 0.50 beside
        # 0.25 and 0.75 rather than as a different quantity.
        return f'{value:.2f}'
    label = f'{value:g}' if isinstance(value, (int, float)) else str(value)
    if key.endswith('HALF_LIFE'):
        return f'{label} Epochs'
    return label


def build_grid(runs, row_key, col_key, spec, number, caption, precision,
               configs=None):
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
                if any(hyperparameter(data, key, name, configs) is not None
                       for key in CONTROL_MARKERS)]
    if controls:
        print(f'  excluding {len(controls)} control run(s) from the grid: '
              f'{", ".join(controls)}')
        runs = [(name, data) for name, data in runs if name not in set(controls)]

    rows = list(reversed(axis_values(runs, row_key, configs)))
    cols = axis_values(runs, col_key, configs)
    arms = axis_values(runs, ARM_KEY, configs) or [None]

    table = Table(number, caption, GRID_STUB.get(row_key, heading(row_key)),
                  [format_axis(value, col_key) for value in cols])
    table.add_footnote(f'Columns are the {GRID_STUB.get(col_key, heading(col_key)).lower()}.')
    table.add_footnote(f'Cells are {metric_heading(spec)}. {MISSING} marks a run with no '
                       f'result on disk.')

    for arm in arms:
        if len(arms) > 1 or arm is not None:
            table.add_category(ARM_HEADINGS.get(arm, str(arm)))
        for row_value in rows:
            cells = []
            for col_value in cols:
                matches = [data for name, data in runs
                           if hyperparameter(data, ARM_KEY, name, configs) == arm
                           and hyperparameter(data, row_key, name, configs) == row_value
                           and hyperparameter(data, col_key, name, configs) == col_value]
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


def flat_columns(runs, keys, configs=None):
    """The configuration columns at least one run carries a value for.

    Args:
        runs: Sequence of (name, parsed YAML).
        keys: Candidate hyperparameters, in the order the table presents them.
        configs: The dict `config_index` returns, or None.

    Returns:
        The subset of `keys` any run has a value for, in the given order. A half-life is kept
        when a run records it as None, since a flat schedule is a value.
    """
    kept = []
    for key in keys:
        if any(key in (data.get('hyperparameters') or {})
               or key in ((configs or {}).get(name) or {})
               or from_name(name, key) is not None
               for name, data in runs):
            kept.append(key)
    return kept


def spec_order(patterns):
    """The sweep order a tuning spec defines: its hyperparameters, and each one's values.

    A one-at-a-time sweep is reported as a block per hyperparameter, and both the order of the
    blocks and the order within one are the spec's, not the values' -- a spec may list a
    weight from strongest to weakest. Without a spec neither is recoverable, since the trial
    names carry only the alias and the value.

    Args:
        patterns: Comma-separated globs for the spec files, or None.

    Returns:
        Dict of hyperparameter to its list of values, in spec order. Insertion order is the
        block order. Specs are merged, so the arms of one phase can be passed together.
    """
    order = {}
    for pattern in (patterns or '').split(','):
        if not pattern.strip():
            continue
        for path in sorted(glob.glob(pattern.strip(), recursive=True)):
            try:
                with open(path) as handle:
                    spec = yaml.safe_load(handle) or {}
            except (OSError, yaml.YAMLError):
                continue
            for key, entry in (spec.get('GRID') or {}).items():
                values = (entry or {}).get('values') if isinstance(entry, dict) else entry
                if isinstance(values, list):
                    order.setdefault(key, list(values))
    return order


def flat_order(runs, keys, configs=None, order=None):
    """Runs in reporting order: by arm, then by the hyperparameter each one varies.

    A one-at-a-time sweep has a centre and one block of variants per hyperparameter. Sorting
    by name interleaves those blocks in whatever order the aliases happen to sort in, so the
    order is rebuilt from the values.

    Args:
        runs: Sequence of (name, parsed YAML).
        keys: The configuration columns, in column order.
        configs: The dict `config_index` returns, or None.
        order: The dict `spec_order` returns. Blocks and the rows within them follow it where
            it reaches; anything it does not name falls back to column order, ascending.

    Returns:
        The runs, reordered.
    """
    arms = axis_values(runs, ARM_KEY, configs) or [None]
    order = order or {}
    blocks = list(order) + [key for key in keys if key not in order]

    # The centre is the value each hyperparameter takes in the most runs of its own arm, which
    # is the setting the other blocks hold it at. Per arm, because the arms reach this phase
    # carrying different inherited settings: a centre pooled over both would read every run of
    # the smaller arm as varying something.
    centre = {}
    for arm in arms:
        for key in keys:
            counts = {}
            for name, data in runs:
                if hyperparameter(data, ARM_KEY, name, configs) != arm:
                    continue
                value = hyperparameter(data, key, name, configs)
                counts[value] = counts.get(value, 0) + 1
            if counts:
                centre[(arm, key)] = max(counts, key=lambda value: counts[value])

    def rank(item):
        name, data = item
        arm = hyperparameter(data, ARM_KEY, name, configs)
        arm_index = arms.index(arm) if arm in arms else len(arms)
        for key in blocks:
            if (arm, key) not in centre:
                continue
            value = hyperparameter(data, key, name, configs)
            if value != centre[(arm, key)]:
                # Sorts after the centre, then by which hyperparameter varies, then by the
                # position the spec gives the value -- descending, where the spec is.
                within = (order[key].index(value) if key in order and value in order[key]
                          else (value if isinstance(value, (int, float)) else 0.0))
                return (arm_index, 1, blocks.index(key), within, name)
        return (arm_index, 0, 0, 0.0, name)

    return sorted(runs, key=rank)


def build_flat(runs, specs, number, caption, precision, configs=None, columns=None,
               highlight=True, order=None):
    """One row per run: the arm, the configuration it ran at, then the metrics.

    The arm is the stub column and the run name is not reported, since the name is an internal
    identifier and the configuration beside it is what identifies the row.

    Args:
        runs: Sequence of (name, parsed YAML).
        specs: The block:metric specs to report.
        number: Table number for the caption.
        caption: Caption text.
        precision: Decimal places for a metric.
        configs: The dict `config_index` returns, or None.
        columns: Configuration columns to report, in order. Defaults to `FLAT_COLUMNS`
            filtered to those the runs carry.
        highlight: Bold the best-scoring row of each arm on the last metric.
        order: The dict `spec_order` returns, or None.

    Returns:
        The assembled `Table`.
    """
    keys = columns if columns is not None else flat_columns(runs, FLAT_COLUMNS, configs)
    ordered = flat_order(runs, keys, configs, order)

    # Footnotes are lettered in call order, which has to be the order they are referenced in:
    # left to right along the header, starting with the stub.
    table = Table(number, caption, heading(ARM_KEY), [])
    stub_note = table.add_footnote(ARM_FOOTNOTE)
    table.stub_head = f'{heading(ARM_KEY)}<sup>{stub_note}</sup>'
    headings = []
    for key in keys:
        text = heading(key)
        if key in COLUMN_FOOTNOTES:
            text += f'<sup>{table.add_footnote(COLUMN_FOOTNOTES[key])}</sup>'
        headings.append(text)
    table.columns = headings + [metric_heading(spec) for spec in specs]

    best = best_rows(ordered, specs[-1], configs) if highlight and specs else set()

    for name, data in ordered:
        cells = []
        for key in keys:
            value = hyperparameter(data, key, name, configs)
            cells.append(MISSING if value is None and not key.endswith('HALF_LIFE')
                         else format_axis(value, key))
        for spec in specs:
            value = metric_value(data, spec)
            cells.append(MISSING if value is None else f'{value:.{precision}f}')
        arm = hyperparameter(data, ARM_KEY, name, configs)
        label = ARM_SHORT.get(arm, render(arm))
        if name in best:
            label = f'<b>{label}</b>'
            cells = [f'<b>{cell}</b>' for cell in cells]
        table.add_row(label, cells)
    return table


def best_rows(runs, spec, configs=None):
    """The name of the best-scoring run in each arm.

    Args:
        runs: Sequence of (name, parsed YAML).
        spec: The block:metric the rows are compared on.
        configs: The dict `config_index` returns, or None.

    Returns:
        Set of experiment names. A loss is best at its minimum and everything else at its
        maximum, so a table of losses is not marked backwards.
    """
    minimize = 'loss' in spec.lower()
    best = {}
    for name, data in runs:
        value = metric_value(data, spec)
        if value is None:
            continue
        arm = hyperparameter(data, ARM_KEY, name, configs)
        current = best.get(arm)
        if current is None or (value < current[1] if minimize else value > current[1]):
            best[arm] = (name, value)
        # A tie keeps the first, which is the earlier row.
    return {name for name, _ in best.values()}


def write_csv(path, table):
    """Write the same numbers as a CSV, one line per row.

    Inline markup is stripped, so a footnote marker or an emphasised row does not reach the
    CSV as tags.
    """
    import csv as csv_module
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv_module.writer(handle)
        writer.writerow([strip_markup(cell)
                         for cell in [table.stub_head] + table.columns])
        group = ''
        for row in table.rows:
            if row.kind == 'category':
                group = row.label
                continue
            cells = ([group] if group else []) + [row.label] + row.cells
            writer.writerow([strip_markup(cell) for cell in cells])
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
    parser.add_argument('--configs', default=DEFAULT_CONFIG_GLOB,
                        help='Recursive glob for the trial configs, read for a '
                             'hyperparameter a run did not record (default: '
                             f'{DEFAULT_CONFIG_GLOB})')
    parser.add_argument('--columns', default=None,
                        help='flat layout: comma-separated configuration columns, in order. '
                             'Defaults to those of FLAT_COLUMNS the runs carry.')
    parser.add_argument('--spec', default=None,
                        help='flat layout: comma-separated globs for the tuning specs, which '
                             'give the order of the blocks and of the rows within them')
    parser.add_argument('--no_highlight', action='store_true',
                        help='flat layout: do not bold the best-scoring row of each arm')
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
    configs = config_index(args.configs) if args.configs else {}

    if args.layout == 'grid':
        spec = args.metric or ('val:Optimization_Loss' if args.task == 'pretrain'
                               else 'val:AUPRC')
        check_metrics(runs, [spec])
        table = build_grid(runs, args.row, args.col, spec, args.table_number, args.caption,
                           args.precision, configs)
    else:
        specs = [item.strip() for item in
                 (args.metrics or 'val:AUROC,val:AUPRC').split(',') if item.strip()]
        check_metrics(runs, specs)
        columns = ([item.strip() for item in args.columns.split(',') if item.strip()]
                   if args.columns else None)
        table = build_flat(runs, specs, args.table_number, args.caption, args.precision,
                           configs, columns, not args.no_highlight, spec_order(args.spec))

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
