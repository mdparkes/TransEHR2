#!/usr/bin/env python3
"""Produce the result tables, one Word document per cohort.

Reads the per-fold prediction CSVs that dump_finetuned_predictions.py writes, computes each
task's metrics, compares every experiment against a nominated control with the corrected
resampled t test of Nadeau & Bengio (2003), controls the false discovery rate with the
Benjamini-Hochberg procedure, and writes a numbered table per task.

The experiments split into two cohorts, each reported against its own in-stay-only control:
every model in a set is compared with the one that reads no pre-admission data at all, which is
the contrast the tables exist to make. Column order and control are properties of the design,
so they are declared below rather than retyped per run.

Usage:
    python report_results_tables.py
    python report_results_tables.py --tasks mortality --cohorts dischargesubset
    python report_results_tables.py --dry_run

    # Any other set of experiments, in the given column order
    python report_results_tables.py --experiments 3 1 2 --control 3 --tasks mortality

Options this does not define are passed through to the per-task reporter, so the threshold,
metric, fold and formatting flags all still apply:

    python report_results_tables.py --tasks mortality --threshold 0.5
    python report_results_tables.py --list-metrics --tasks phenotype

This replaces the former per-task entry points. What each task reports lives in
reporting/tasks.py; the tables and statistics are unchanged.
"""

import argparse
import os
import sys

from reporting.cli import build_parser, run
from reporting.tasks import TASK_SPECS, TASKS


# (key, caption suffix, experiment numbers in column order, control)
COHORTS = (
    ('dischargesubset',
     'patients with at least one pre-admission discharge summary',
     (10, 11, 12, 13, 14), 10),
    ('historysubset',
     'patients with at least one pre-admission record',
     (15, 16, 17), 15),
)

COHORT_KEYS = tuple(key for key, *_ in COHORTS)


def report_one(task_spec, argv):
    """Build and run one task's table.

    Args:
        task_spec: The `TaskSpec` for this task.
        argv: Arguments for the per-task parser.

    Returns:
        Process exit status from the reporter.
    """
    parser = build_parser(
        task_spec.key,
        description=__doc__.split('\n\n')[0],
        default_caption=f'{task_spec.caption} evaluation results.',
        classification=task_spec.classification,
        phenotype=task_spec.phenotype,
    )
    args = parser.parse_args(argv)
    if task_spec.default_metrics and not args.metrics and not args.list_metrics:
        args.metrics = list(task_spec.default_metrics)
    return run(args, task_spec.key, task_spec.specs, task_spec.threshold_note)


def selected_cohorts(args):
    """The (key, caption, experiments, control) groups this run reports."""
    if args.experiments:
        if args.control is None:
            raise SystemExit('--experiments needs --control naming the reference column.')
        if args.control not in args.experiments:
            raise SystemExit(
                f'--control {args.control} is not among --experiments '
                f'{" ".join(str(n) for n in args.experiments)}; the control is one of the '
                f'columns, not a separate model.'
            )
        return [('custom', None, tuple(args.experiments), args.control)]
    return [group for group in COHORTS if group[0] in args.cohorts]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog='Unrecognised options are passed through to the per-task reporter.',
    )
    parser.add_argument('--tables_dir', default='tables',
                        help='Directory for the generated documents (default: tables)')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS),
                        help='Tasks to report (default: all three)')
    parser.add_argument('--cohorts', nargs='+', choices=COHORT_KEYS, default=list(COHORT_KEYS),
                        help='Cohorts to report (default: both)')
    parser.add_argument('--experiments', nargs='+', type=int, default=None,
                        help='Report these experiments instead of a declared cohort, in the '
                             'order their columns should appear')
    parser.add_argument('--control', type=int, default=None,
                        help='Experiment every other column is tested against; required with '
                             '--experiments')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be reported without running it')
    args, passthrough = parser.parse_known_args(argv)

    os.makedirs(args.tables_dir, exist_ok=True)
    status, missing = 0, []

    for key, cohort_caption, experiments, control in selected_cohorts(args):
        output = os.path.join(args.tables_dir, f'{key}_tables.docx')
        if not args.dry_run and os.path.exists(output):
            # Tables are appended, so a stale document would grow rather than be replaced.
            os.remove(output)
        written = 0
        for number, task in enumerate(args.tasks, start=1):
            task_spec = TASK_SPECS[task]
            caption = (f'{task_spec.caption} results, {cohort_caption}.' if cohort_caption
                       else f'{task_spec.caption} evaluation results.')
            task_argv = [
                '--experiments', *[str(n) for n in experiments],
                '--control', str(control),
                '--table-number', str(number),
                '--caption', caption,
                '--output', output,
                '--stats-csv', os.path.join(args.tables_dir, f'{key}_{task}_stats.csv'),
                *(['--append'] if written else []),
                *passthrough,
            ]

            print(f'\n=== {key}: {task} ===')
            if args.dry_run:
                print(' '.join(task_argv))
                continue
            try:
                status = report_one(task_spec, task_argv) or status
            except SystemExit as exc:
                # A task whose predictions are not dumped yet exits with the path it wanted.
                # Name it and carry on: the tables already built are still worth having, and
                # the remaining tasks may well be ready.
                print(f'  skipped: {exc}', file=sys.stderr)
                missing.append(f'{key}/{task}')
                continue
            written += 1

    if args.dry_run:
        print('\nDry run: nothing written.')
        return status
    print(f'\nDocuments and statistics written to {args.tables_dir}/')
    if missing:
        print(f'Not reported, no predictions dumped: {", ".join(missing)}', file=sys.stderr)
        status = status or 1
    return status


if __name__ == '__main__':
    sys.exit(main())
