#!/usr/bin/env python3
"""Produce both sets of revision tables, one document per cohort.

The revision's experiments split into two cohorts, and each is reported against its own
in-stay-only control: everything in a set is compared with the model that reads no
pre-admission data at all, which is the contrast the tables exist to make. Column order and
control are properties of the design, so they live here rather than in retyped command lines.

Each cohort gets one Word document holding a numbered table per task, in the order mortality,
length of stay, phenotype. Metrics, the corrected resampled t test across folds, and the
Benjamini-Hochberg correction all come from the per-task reporters unchanged.

Usage:
    python report_revision_tables.py
    python report_revision_tables.py --tables_dir tables --split test
    python report_revision_tables.py --dry_run

Requires the per-fold prediction CSVs that dump_finetuned_predictions.py writes. The per-task
reporters read those rather than the evaluation YAMLs, because a paired test over folds needs
per-episode predictions and not one aggregate score per fold.
"""

import argparse
import os
import sys

import report_length_of_stay
import report_mortality
import report_phenotype


# (key, caption suffix, experiment numbers in column order, control)
COHORTS = (
    ('dischargesubset',
     'patients with at least one pre-admission discharge summary',
     (10, 11, 12, 13, 14), 10),
    ('historysubset',
     'patients with at least one pre-admission record',
     (15, 16, 17), 15),
)

# (task key, reporter module, caption stem)
TASKS = (
    ('mortality', report_mortality, 'In-hospital mortality'),
    ('length_of_stay', report_length_of_stay, 'Length of stay'),
    ('phenotype', report_phenotype, 'Phenotype classification'),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tables_dir', default='tables',
                        help='Directory for the generated documents (default: tables)')
    parser.add_argument('--model_dir', default='./models',
                        help='Directory holding one subdirectory per experiment')
    parser.add_argument('--split', default='test', help='Data split to report')
    parser.add_argument('--folds', nargs='+', default=None,
                        help='Restrict every experiment to these folds')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the invocations without running them')
    args = parser.parse_args(argv)

    os.makedirs(args.tables_dir, exist_ok=True)
    status = 0
    missing = []
    for key, cohort_caption, experiments, control in COHORTS:
        output = os.path.join(args.tables_dir, f'{key}_tables.docx')
        if not args.dry_run and os.path.exists(output):
            # Tables are appended, so a stale document would grow rather than be replaced.
            os.remove(output)
        for number, (task, reporter, stem) in enumerate(TASKS, start=1):
            argv_task = [
                '--experiments', *[str(n) for n in experiments],
                '--control', str(control),
                '--model-dir', args.model_dir,
                '--split', args.split,
                '--table-number', str(number),
                '--caption', f'{stem} results, {cohort_caption}.',
                '--output', output,
                '--stats-csv', os.path.join(args.tables_dir, f'{key}_{task}_stats.csv'),
            ]
            if number > 1:
                argv_task.append('--append')
            if args.folds:
                argv_task += ['--folds', *args.folds]

            print(f"\n=== {key}: {task} ===")
            if args.dry_run:
                print(f"python report_{task}.py " + ' '.join(argv_task))
                continue
            try:
                result = reporter.main(argv_task)
            except SystemExit as exc:
                # A task whose predictions have not been dumped yet exits with the path it
                # wanted. Report it and carry on: the tasks that are ready are still worth
                # having, and aborting here would discard tables already built.
                print(f'  skipped: {exc}', file=sys.stderr)
                missing.append(f'{key}/{task}')
                continue
            status = status or (result or 0)

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
