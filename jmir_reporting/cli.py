"""Shared command-line machinery for the per-task reporting scripts.

Each task has its own entry point (``report_mortality.py``,
``report_length_of_stay.py``, ``report_phenotype.py``) because the tasks
differ in which metrics they report and how those metrics are grouped.
Everything the three have in common lives here: argument parsing,
resolving experiment numbers to column headings, running the corrected
resampled t tests, controlling the false discovery rate, and assembling
the result into a :class:`~jmir_reporting.tables.Table`.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import yaml

from .evaluation import evaluate_experiment
from .formatting import fmt_cell, fmt_number, fmt_p_value, fmt_t_statistic
from .stats import (benjamini_hochberg, corrected_resampled_ttest,
                    mean_of_folds, standard_error_of_mean)
from .tables import Table, build_document, render_text, strip_markup

DEFAULT_LABEL_FILE = 'reporting_labels.yaml'

F1_LABEL = '<i>F</i><sub>1</sub>-score'


class MetricSpec:
    """How one metric is presented in a table.

    Attributes:
        key: The metric name as produced by
            :mod:`jmir_reporting.evaluation`.
        label: The row heading, in sentence case, which may contain
            inline markup.
        level: ``0`` for a top-level row, ``1`` for a row beneath a
            category heading.
        compare: Whether to test this metric against the control. Set to
            ``False`` for quantities such as prevalence that describe the
            data split rather than the model.
        precision: Decimal places, overriding the table default.
    """

    __slots__ = ('key', 'label', 'level', 'compare', 'precision')

    def __init__(self, key, label, level=0, compare=True, precision=None):
        self.key = key
        self.label = label
        self.level = level
        self.compare = compare
        self.precision = precision


class CategorySpec:
    """A bold grouping heading between metric rows.

    Attributes:
        label: The category heading, e.g. ``'Microaverages'``.
    """

    __slots__ = ('label',)

    def __init__(self, label):
        self.label = label


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def threshold_argument(value):
    """Parse the ``--threshold`` argument.

    Args:
        value: The raw string from the command line.

    Returns:
        The string ``'prevalence'`` or a float in [0, 1].

    Raises:
        argparse.ArgumentTypeError: If the value is neither.
    """
    if value == 'prevalence':
        return value
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{value!r} is neither "prevalence" nor a number'
        ) from None
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(
            f'A fixed threshold must lie in [0, 1], got {number}'
        )
    return number


def build_parser(task, description, default_caption,
                 classification=True, phenotype=False):
    """Build the argument parser for one task's reporting script.

    Args:
        task: The task name, used in help text and defaults.
        description: The parser description.
        default_caption: Default table caption.
        classification: Whether to expose the threshold options.
        phenotype: Whether to expose the per-label threshold scope.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Not marked required, so that --list-metrics works on its own;
    # validate_args() enforces them for a real run.
    parser.add_argument(
        '--experiments', type=int, nargs='+', default=None,
        metavar='N',
        help='Experiment numbers, in the order their columns should '
             'appear in the table (required)'
    )
    parser.add_argument(
        '--control', type=int, default=None, metavar='N',
        help='Experiment number of the control that every other column '
             'is tested against (required)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='./models',
        help='Directory holding one subdirectory per experiment'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        help='Data split to report'
    )
    parser.add_argument(
        '--folds', type=str, nargs='+', default=None, metavar='FOLD',
        help='Fold directory names to use; default is to auto-discover '
             'every fold except fold0'
    )
    parser.add_argument(
        '--precision', type=int, default=3,
        help='Decimal places for means and standard errors'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help="Significance level, used to decide when a P value needs a "
             "third decimal place"
    )
    parser.add_argument(
        '--fdr-scope', choices=('table', 'row', 'none'), default='table',
        help='Family over which the Benjamini-Hochberg procedure '
             'controls the false discovery rate: every comparison in '
             'the table, every comparison within one metric row, or no '
             'correction'
    )
    parser.add_argument(
        '--show-raw-p', action='store_true',
        help='Report the uncorrected P value alongside the adjusted one'
    )
    parser.add_argument(
        '--table-number', type=int, default=1,
        help='Table number used in the caption'
    )
    parser.add_argument(
        '--caption', type=str, default=default_caption,
        help='Table caption, without the "Table N." prefix'
    )
    parser.add_argument(
        '--labels', type=str, default=None,
        help=f'YAML file mapping experiment numbers to column headings '
             f'(default: {DEFAULT_LABEL_FILE} beside this script)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path of the Word document to write; omit to print only'
    )
    parser.add_argument(
        '--append', action='store_true',
        help='Append to the Word document at --output instead of '
             'replacing it, so several tables can be collected into one '
             'file'
    )
    parser.add_argument(
        '--stats-csv', type=str, default=None,
        help='Optional CSV path for the full per-fold values and test '
             'statistics'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress the statistical detail block on stdout'
    )
    parser.add_argument(
        '--metrics', type=str, nargs='+', default=None, metavar='KEY',
        help='Report only these metrics, in this order; default is the '
             'full set for the task. Use --list-metrics to see the keys'
    )
    parser.add_argument(
        '--list-metrics', action='store_true',
        help='Print the available metric keys and exit'
    )

    if classification:
        parser.add_argument(
            '--threshold', type=threshold_argument, default='prevalence',
            help='Decision threshold: "prevalence" calibrates it per '
                 'fold on the calibration split so that the predicted '
                 'positive rate matches the observed prevalence, or give '
                 'a fixed number such as 0.5'
        )
        parser.add_argument(
            '--calibration-split', type=str, default='val',
            help='Split on which to calibrate the threshold; never the '
                 'split being reported'
        )
    if phenotype:
        parser.add_argument(
            '--phenotype-threshold-scope',
            choices=('per-label', 'global'), default='per-label',
            help='Calibrate one threshold per label, or a single '
                 'threshold shared by every label'
        )

    return parser


def validate_args(args):
    """Check the mutually dependent arguments.

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: If either argument is missing, if the control is not
            among the reported experiments, or if an experiment number is
            repeated.
    """
    missing = [name for name in ('experiments', 'control')
               if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            'Missing required argument(s): '
            + ', '.join(f'--{name}' for name in missing)
        )
    if args.control not in args.experiments:
        raise SystemExit(
            f'--control {args.control} must be one of --experiments '
            f'{" ".join(str(n) for n in args.experiments)}'
        )
    duplicates = {n for n in args.experiments
                  if args.experiments.count(n) > 1}
    if duplicates:
        raise SystemExit(
            f'Repeated experiment numbers: '
            f'{", ".join(str(n) for n in sorted(duplicates))}'
        )


# ------------------------------------------------------------------
# Column headings
# ------------------------------------------------------------------

def load_labels(path=None):
    """Load the mapping from experiment number to column heading.

    Args:
        path: Explicit path to a YAML file, or ``None`` to look for
            ``reporting_labels.yaml`` in the current directory and then
            beside the package.

    Returns:
        A dict mapping ``int`` experiment number to heading string.
        Empty if no file was found.
    """
    candidates = []
    if path:
        candidates.append(path)
    else:
        candidates.append(DEFAULT_LABEL_FILE)
        candidates.append(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            DEFAULT_LABEL_FILE
        ))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate) as handle:
                raw = yaml.safe_load(handle) or {}
            return {int(k): str(v)
                    for k, v in (raw.get('columns') or raw).items()}

    if path:
        raise SystemExit(f'Label file not found: {path}')
    return {}


def column_heading(number, labels, result):
    """Choose the column heading for one experiment.

    Args:
        number: The experiment number.
        labels: Mapping from number to heading.
        result: The :class:`~jmir_reporting.evaluation.ExperimentResult`,
            used as a fallback.

    Returns:
        The heading string.
    """
    if number in labels:
        return labels[number]
    return f'Experiment {number} ({result.name})'


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

class Comparison:
    """A single experiment-versus-control comparison for one metric.

    Attributes:
        metric: The metric key.
        number: The experiment number.
        mean: Mean across folds.
        se: Standard error of the mean.
        folds: Per-fold values.
        test: The :class:`~jmir_reporting.stats.TestResult`, or ``None``
            for the control column and for non-comparable metrics.
        p_adjusted: The false-discovery-rate-adjusted P value, or
            ``None``.
    """

    __slots__ = ('metric', 'number', 'mean', 'se', 'folds', 'test',
                 'p_adjusted')

    def __init__(self, metric, number, mean, se, folds, test):
        self.metric = metric
        self.number = number
        self.mean = mean
        self.se = se
        self.folds = folds
        self.test = test
        self.p_adjusted = None

    @property
    def p_reported(self):
        """The P value that goes in the table, adjusted where available."""
        if self.test is None:
            return None
        if self.p_adjusted is not None:
            return self.p_adjusted
        return self.test.p_value


def compare_experiments(results, order, control, metric_specs, fdr_scope):
    """Run every comparison for a table and adjust the P values.

    Args:
        results: Mapping from experiment number to
            :class:`~jmir_reporting.evaluation.ExperimentResult`.
        order: Experiment numbers in column order.
        control: The control experiment number.
        metric_specs: The :class:`MetricSpec` objects to report.
        fdr_scope: ``'table'``, ``'row'`` or ``'none'``.

    Returns:
        A dict mapping ``(metric_key, experiment_number)`` to
        :class:`Comparison`.
    """
    comparisons = {}
    control_result = results[control]

    for spec in metric_specs:
        for number in order:
            result = results[number]
            try:
                values = result.values(spec.key)
            except KeyError:
                raise SystemExit(
                    f'experiment {number} ({result.name}) has no metric '
                    f'{spec.key!r}; available: '
                    f'{", ".join(sorted(result.metrics))}'
                ) from None
            test = None
            if number != control and spec.compare:
                test = corrected_resampled_ttest(
                    values, control_result.values(spec.key)
                )
            comparisons[(spec.key, number)] = Comparison(
                spec.key, number, mean_of_folds(values),
                standard_error_of_mean(values), values, test
            )

    if fdr_scope == 'none':
        return comparisons

    if fdr_scope == 'row':
        families = [
            [(spec.key, n) for n in order if n != control]
            for spec in metric_specs
        ]
    else:
        families = [[
            (spec.key, n)
            for spec in metric_specs for n in order if n != control
        ]]

    for family in families:
        keys = [k for k in family if comparisons[k].test is not None]
        if not keys:
            continue
        raw = [comparisons[k].test.p_value for k in keys]
        for key, adjusted in zip(keys, benjamini_hochberg(raw)):
            comparisons[key].p_adjusted = float(adjusted)

    return comparisons


# ------------------------------------------------------------------
# Table assembly
# ------------------------------------------------------------------

def build_table(args, results, order, control, specs, labels,
                threshold_note=None):
    """Assemble the manuscript table.

    Footnotes are registered in the order JMIR requires, which is left to
    right and then top to bottom: the notes attached to the stub heading
    come first, then any note attached to a row further down.

    Args:
        args: The parsed arguments.
        results: Mapping from experiment number to
            :class:`~jmir_reporting.evaluation.ExperimentResult`.
        order: Experiment numbers in column order.
        control: The control experiment number.
        specs: A sequence of :class:`MetricSpec` and :class:`CategorySpec`
            objects, in row order.
        labels: Mapping from experiment number to column heading.
        threshold_note: Optional sentence describing how the decision
            threshold was chosen.

    Returns:
        A tuple ``(table, comparisons)``.
    """
    metric_specs = [s for s in specs if isinstance(s, MetricSpec)]
    comparisons = compare_experiments(
        results, order, control, metric_specs, args.fdr_scope
    )

    n_folds = len(results[control].folds)
    columns = [column_heading(n, labels, results[n]) for n in order]

    short = [f'Expt {n}' + (' (control)' if n == control else '')
             for n in order]
    table = Table(args.table_number, args.caption, 'Metric', columns,
                  short_columns=short)

    markers = []
    markers.append(table.add_footnote(
        f' Values are the mean across the {n_folds} cross-validation '
        f'folds, with the standard error of the mean in parentheses.'
    ))
    markers.append(table.add_footnote(_p_value_footnote(
        args, n_folds, column_heading(control, labels, results[control])
    )))
    if threshold_note:
        markers.append(table.add_footnote(' ' + threshold_note))

    table.stub_head = f'Metric<sup>{",".join(markers)}</sup>'

    non_comparable = [s for s in metric_specs if not s.compare]
    non_comparable_marker = None
    if non_comparable:
        non_comparable_marker = table.add_footnote(
            ' This quantity is a property of the data split rather than '
            'of the model, so it is identical in every column and no '
            'comparison is reported.'
        )

    for spec in specs:
        if isinstance(spec, CategorySpec):
            table.add_category(spec.label)
            continue

        precision = (spec.precision if spec.precision is not None
                     else args.precision)
        cells = []
        for number in order:
            comparison = comparisons[(spec.key, number)]
            cells.append(_format_comparison(
                comparison, precision, args
            ))

        label = spec.label
        if not spec.compare and non_comparable_marker:
            label += f'<sup>{non_comparable_marker}</sup>'
        table.add_row(label, cells, level=spec.level)

    return table, comparisons


def _format_comparison(comparison, precision, args):
    """Render one table cell.

    Args:
        comparison: The :class:`Comparison` for this cell.
        precision: Decimal places for the mean and SE.
        args: The parsed arguments.

    Returns:
        The cell text.
    """
    if comparison.test is None:
        return fmt_cell(comparison.mean, comparison.se, None, precision,
                        args.alpha)

    if math.isnan(comparison.test.p_value):
        # The test was not applicable, e.g. the two arms were identical in
        # every fold. Report the value without a P value.
        return fmt_cell(comparison.mean, comparison.se, None, precision,
                        args.alpha)

    body = f'SE {fmt_number(comparison.se, precision)}'
    if args.show_raw_p and comparison.p_adjusted is not None:
        body += (f'; {fmt_p_value(comparison.test.p_value, args.alpha)}'
                 f'; adjusted '
                 f'{fmt_p_value(comparison.p_adjusted, args.alpha)}')
    else:
        body += f'; {fmt_p_value(comparison.p_reported, args.alpha)}'
    return f'{fmt_number(comparison.mean, precision)} ({body})'


def _p_value_footnote(args, n_folds, control_label):
    """Compose the footnote describing the statistical test.

    Args:
        args: The parsed arguments.
        n_folds: Number of cross-validation folds.
        control_label: Column heading of the control.

    Returns:
        The footnote text.
    """
    text = (
        f' <i>P</i> values are from 2-tailed corrected resampled '
        f'<i>t</i> tests against the "{strip_markup(control_label)}" '
        f'model on {n_folds - 1} <i>df</i>. The correction of Nadeau and '
        f'Bengio inflates the variance of the mean per-fold difference by '
        f'the ratio of test set size to training set size, fixed at '
        f'1/({n_folds} − 1) for a single run of '
        f'{n_folds}-fold cross-validation.'
    )
    if args.fdr_scope == 'table':
        text += (' Reported values are adjusted by the '
                 'Benjamini-Hochberg procedure across every comparison '
                 'in this table.')
    elif args.fdr_scope == 'row':
        text += (' Reported values are adjusted by the '
                 'Benjamini-Hochberg procedure across the comparisons '
                 'within each metric.')
    else:
        text += ' Values are not adjusted for multiple comparisons.'
    if args.show_raw_p and args.fdr_scope != 'none':
        text += (' The unadjusted value is given first and the adjusted '
                 'value second.')
    return text


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def print_statistical_detail(comparisons, order, control, metric_specs,
                             results, alpha):
    """Print the test statistics that the table itself omits.

    The Word table carries only P values, so the t statistics, the mean
    per-fold differences and the unadjusted P values are printed here for
    the record.

    Args:
        comparisons: Output of :func:`compare_experiments`.
        order: Experiment numbers in column order.
        control: The control experiment number.
        metric_specs: The reported :class:`MetricSpec` objects.
        results: Mapping from experiment number to
            :class:`~jmir_reporting.evaluation.ExperimentResult`.
        alpha: Significance level.
    """
    print()
    print('Statistical detail (corrected resampled t tests versus '
          f'experiment {control})')
    print('=' * 78)
    header = f'{"Metric":<28}{"Expt":>5}  {"Difference":>11}  ' \
             f'{"t (df)":>12}  {"P":>10}  {"P adjusted":>11}'
    print(header)
    print('-' * len(header))

    for spec in metric_specs:
        for number in order:
            if number == control:
                continue
            comparison = comparisons[(spec.key, number)]
            if comparison.test is None:
                continue
            test = comparison.test
            adjusted = ('—' if comparison.p_adjusted is None
                        else fmt_p_value(comparison.p_adjusted, alpha))
            print(f'{spec.key:<28}{number:>5}  '
                  f'{fmt_number(test.mean_difference, 4):>11}  '
                  f'{fmt_t_statistic(test.statistic, test.df):>12}  '
                  f'{fmt_p_value(test.p_value, alpha):>10}  '
                  f'{adjusted:>11}')
            if test.note:
                print(f'{"":<33}  note: {test.note}')

    print()
    print('Folds used:')
    for number in order:
        result = results[number]
        summary = result.threshold_summary()
        line = f'  experiment {number} ({result.name}): ' \
               f'{", ".join(result.folds)}'
        if summary:
            line += f'; threshold {summary}'
        print(line)


def write_stats_csv(path, comparisons, order, control, metric_specs,
                    results):
    """Write every per-fold value and test statistic to a CSV.

    Args:
        path: Destination CSV path.
        comparisons: Output of :func:`compare_experiments`.
        order: Experiment numbers in column order.
        control: The control experiment number.
        metric_specs: The reported :class:`MetricSpec` objects.
        results: Mapping from experiment number to
            :class:`~jmir_reporting.evaluation.ExperimentResult`.

    Returns:
        The path written.
    """
    fold_names = results[control].folds
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['metric', 'experiment', 'experiment_dir', 'is_control',
             'mean', 'sem']
            + [f'fold_{name}' for name in fold_names]
            + ['mean_difference', 't_statistic', 'df', 'p_value',
               'p_value_bh_adjusted', 'note']
        )
        for spec in metric_specs:
            for number in order:
                comparison = comparisons[(spec.key, number)]
                test = comparison.test
                writer.writerow(
                    [spec.key, number, results[number].name,
                     int(number == control),
                     comparison.mean, comparison.se]
                    + list(comparison.folds)
                    + [
                        '' if test is None else test.mean_difference,
                        '' if test is None else test.statistic,
                        '' if test is None else test.df,
                        '' if test is None else test.p_value,
                        '' if comparison.p_adjusted is None
                        else comparison.p_adjusted,
                        '' if test is None or not test.note else test.note,
                    ]
                )
    return path


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def select_metrics(specs, keys):
    """Restrict and reorder the reported rows.

    Category headings that end up with no rows beneath them are dropped.

    Args:
        specs: The full sequence of :class:`MetricSpec` and
            :class:`CategorySpec` objects for the task.
        keys: Metric keys to keep, in the desired order, or ``None`` to
            keep everything.

    Returns:
        The filtered sequence.

    Raises:
        SystemExit: If a requested key is not defined for the task.
    """
    if not keys:
        return list(specs)

    by_key = {s.key: s for s in specs if isinstance(s, MetricSpec)}
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        raise SystemExit(
            f'Unknown metric(s): {", ".join(unknown)}. Available: '
            f'{", ".join(by_key)}'
        )

    # Preserve the category a metric belongs to, following the order the
    # user asked for.
    category_of = {}
    current = None
    for spec in specs:
        if isinstance(spec, CategorySpec):
            current = spec.label
        else:
            category_of[spec.key] = current

    selected = []
    emitted = None
    for key in keys:
        category = category_of[key]
        if category != emitted and category is not None:
            selected.append(CategorySpec(category))
            emitted = category
        elif category is None:
            emitted = None
        selected.append(by_key[key])
    return selected


def list_metrics(specs):
    """Print the metric keys available for a task.

    Args:
        specs: The task's full sequence of specs.
    """
    current = None
    for spec in specs:
        if isinstance(spec, CategorySpec):
            current = spec.label
            print(f'{current}:')
            continue
        prefix = '  ' if current else ''
        print(f'{prefix}{spec.key:<32}{strip_markup(spec.label)}')


def run(args, task, specs, threshold_note_builder=None):
    """Evaluate the experiments, print the table and write the outputs.

    Args:
        args: The parsed arguments.
        task: The task name.
        specs: A sequence of :class:`MetricSpec` and :class:`CategorySpec`
            objects, in row order.
        threshold_note_builder: Optional callable taking the results dict
            and returning the threshold footnote sentence.

    Returns:
        Exit status: 0 on success.
    """
    if args.list_metrics:
        list_metrics(specs)
        return 0

    validate_args(args)
    specs = select_metrics(specs, args.metrics)
    labels = load_labels(args.labels)

    threshold_mode = getattr(args, 'threshold', None)
    scope = getattr(args, 'phenotype_threshold_scope', 'per-label')

    results = {}
    for number in args.experiments:
        try:
            results[number] = evaluate_experiment(
                args.model_dir, number, task, args.split,
                threshold_mode=threshold_mode,
                calibration_split=getattr(args, 'calibration_split', 'val'),
                threshold_scope=scope,
                folds=args.folds,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f'experiment {number}: {exc}')

    fold_counts = {n: len(r.folds) for n, r in results.items()}
    if len(set(fold_counts.values())) > 1:
        detail = ', '.join(f'{n}: {c}' for n, c in fold_counts.items())
        raise SystemExit(
            f'The corrected resampled t test compares models across the '
            f'same folds, but the fold counts differ ({detail}). Pass '
            f'--folds to restrict every experiment to a common set.'
        )

    threshold_note = (threshold_note_builder(args, results)
                      if threshold_note_builder else None)

    table, comparisons = build_table(
        args, results, args.experiments, args.control, specs, labels,
        threshold_note
    )

    render_text(table)

    metric_specs = [s for s in specs if isinstance(s, MetricSpec)]
    if not args.quiet:
        print_statistical_detail(
            comparisons, args.experiments, args.control, metric_specs,
            results, args.alpha
        )

    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        build_document(table, args.output, append=args.append)
        print(f'\nWrote {args.output}', file=sys.stderr)

    if args.stats_csv:
        write_stats_csv(
            args.stats_csv, comparisons, args.experiments, args.control,
            metric_specs, results
        )
        print(f'Wrote {args.stats_csv}', file=sys.stderr)

    return 0


def describe_threshold(args, results):
    """Compose the footnote sentence describing the decision threshold.

    Args:
        args: The parsed arguments.
        results: Mapping from experiment number to
            :class:`~jmir_reporting.evaluation.ExperimentResult`.

    Returns:
        The sentence, or ``None`` when the task has no threshold.
    """
    mode = getattr(args, 'threshold', None)
    if mode is None:
        return None

    if mode != 'prevalence':
        return (f'Hard class labels were obtained with a fixed decision '
                f'threshold of {fmt_number(float(mode), 2)}.')

    scope = getattr(args, 'phenotype_threshold_scope', None)
    scope_text = ''
    if scope == 'per-label':
        scope_text = ' Thresholds were calibrated separately for each label.'
    elif scope == 'global':
        scope_text = (' A single threshold was shared by every label, '
                      'calibrated on the pooled labels.')

    summaries = '; '.join(
        f'experiment {n} {results[n].threshold_summary()}'
        for n in args.experiments
        if results[n].threshold_summary()
    )

    return (
        f'Within each fold the decision threshold was set on the '
        f'{args.calibration_split} split so that the predicted positive '
        f'rate matched the observed prevalence, and was then applied '
        f'unchanged to the {args.split} split; the threshold was never '
        f'chosen on the {args.split} split, and the same rule was '
        f'applied to every model.{scope_text} Thresholds used: '
        f'{summaries}.'
    )
