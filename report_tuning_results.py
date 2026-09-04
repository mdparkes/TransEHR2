#!/usr/bin/env python3
"""Report the state and the results of a hyperparameter sweep.

Reads the evaluation YAMLs the trials wrote, ranks each hyperparameter's tested values by the
criterion its own grid entry names, and writes the tables. Safe to run at any point: trials
that have not finished are reported as pending rather than treated as missing data, so this
doubles as the progress check while the sweep is in flight.

Usage:
    # Where is the sweep up to?
    python report_tuning_results.py <manifest> --progress

    # Full report: rankings to the terminal, tables to tables/
    python report_tuning_results.py <manifest>

    # Just the tables, no terminal output
    python report_tuning_results.py <manifest> --quiet --docx tables/phase2_tuning.docx

Two selection criteria are in play and which applies is a property of the hyperparameter, not
a choice made here. Learning rate and decay leave the pretraining objective intact, so
pretraining validation loss ranks them. The masking ratios and the time weight rescale that
objective -- a model masked at 0.75 is not solving the same problem as one masked at 0.25 --
so ranking them on pretraining loss would reward the easiest task, and they are selected on
mortality validation performance instead.
"""

import argparse
import os
import sys

from hp_tuning.reporting import format_grid_value, format_metric, print_table, write_tables
from hp_tuning.results import compare_arms, progress, rank_cells, rank_hyperparameter
from hp_tuning.spec import load_manifest


def print_progress(manifest):
    """Print how many trials in each stage have usable results.

    Args:
        manifest: A loaded manifest.

    Returns:
        True if every stage is complete.
    """
    summary = progress(manifest)
    print("Sweep progress")
    print("-" * 70)
    complete = True
    for stage, counts in summary.items():
        print(f"  {stage:<10} {counts['done']:>3} / {counts['total']:<3} complete")
        if counts['pending']:
            complete = False
            for name in counts['pending']:
                print(f"             pending: {name}")
    print()
    return complete


def print_cell_ranking(ranking):
    """Print one arm's cells, best first, for a factorial sweep.

    Coordinates are shown in full because a cell is not identified by any one of them: every
    value appears in several cells, and it is the combination that was measured.

    Args:
        ranking: A ranking dict as returned by :func:`hp_tuning.results.rank_cells`.
    """
    direction = 'lowest' if ranking['direction'] == 'min' else 'highest'
    print(f"\n  Factorial cells, selected on {ranking['criterion']}: "
          f"{direction} {ranking['metric']}")
    usable = [r for r in ranking['results'] if r.is_usable]
    pending = [r for r in ranking['results'] if not r.is_usable]
    ordered = sorted(usable, key=lambda r: r.value, reverse=ranking['direction'] == 'max')
    for result in ordered + pending:
        marker = '  <-- selected' if (
            ranking['best'] is not None and result is ranking['best']
        ) else ''
        shown = format_metric(result.value) if result.is_usable else f'[{result.status}]'
        coordinates = '  '.join(
            f'{name}={format_grid_value(value)}' for name, value in result.grid_value.items()
        )
        print(f"      {coordinates:<60} {shown:>12}{marker}")
        if not result.is_usable:
            print(f"        {result.detail}")
    if not ranking['complete']:
        print("      NOTE incomplete: the selection above is the best of the cells that")
        print("           finished, which is a weaker claim than the best of the grid.")


def print_rankings(manifest):
    """Print the ranking of every hyperparameter on every arm.

    Args:
        manifest: A loaded manifest.
    """
    for arm in manifest['arms']:
        print("=" * 70)
        print(f"Arm: {arm}")
        print("=" * 70)
        if manifest.get('design') == 'factorial':
            print_cell_ranking(rank_cells(manifest, arm))
            print()
            continue
        rows, notes = [], []
        for hyperparameter in manifest['grid']:
            ranking = rank_hyperparameter(manifest, arm, hyperparameter)
            direction = 'lowest' if ranking['direction'] == 'min' else 'highest'
            for index, result in enumerate(ranking['results']):
                selected = ranking['best'] is not None and result is ranking['best']
                shown = format_metric(result.value) if result.is_usable \
                    else f'[{result.status}]'
                rows.append([
                    hyperparameter if index == 0 else '',
                    format_grid_value(result.grid_value),
                    shown,
                    '<-- selected' if selected else '',
                    f"{ranking['criterion']}, {direction} {ranking['metric']}"
                    if index == 0 else '',
                ])
                if not result.is_usable:
                    notes.append(f"  {hyperparameter}="
                                 f"{format_grid_value(result.grid_value)}: {result.detail}")
            if not ranking['complete']:
                notes.append(f"  {hyperparameter}: incomplete, so the selection is the best "
                             f"of the trials that finished")
        print()
        print_table(rows, ['hyperparameter', 'value', 'metric', '', 'ranked on'], indent='  ')
        for note in notes:
            print(note)
        print()

    if len(manifest['arms']) > 1:
        comparison = compare_arms(manifest)
        print("=" * 70)
        print("Encoding arms, head to head at the shared all-defaults centre")
        print("=" * 70)
        direction = 'lowest' if comparison['direction'] == 'min' else 'highest'
        print(f"  selected on {comparison['criterion']}: {direction} {comparison['metric']}")
        for result in comparison['results']:
            marker = '  <-- selected' if (
                comparison['best'] is not None and result is comparison['best']
            ) else ''
            shown = format_metric(result.value) if result.is_usable else f'[{result.status}]'
            print(f"      {str(result.grid_value):>10}  {shown:>12}{marker}")
        print()


def main(argv=None):
    """Report a sweep.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        0 if the report was produced, 1 if --require_complete was given and it is not.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('manifest', type=str, help='Manifest written by generate_tuning_configs.py')
    parser.add_argument('--progress', action='store_true',
                        help='Print only the progress summary, no rankings and no tables')
    parser.add_argument('--tables_dir', type=str, default='tables',
                        help='Directory for the generated tables. Default tables/')
    parser.add_argument('--docx', type=str, default=None,
                        help='Word output path. Defaults to <tables_dir>/<spec>_tuning.docx')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV output path. Defaults to <tables_dir>/<spec>_tuning.csv')
    parser.add_argument('--no_tables', action='store_true', help='Skip writing tables')
    parser.add_argument('--quiet', action='store_true',
                        help='Write the tables without printing anything to the terminal')
    parser.add_argument('--require_complete', action='store_true',
                        help='Exit non-zero if any trial has not produced a usable result. Use '
                             'this in a dependent job so the pipeline stops rather than '
                             'selecting from a partial sweep.')
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    complete = print_progress(manifest) if not args.quiet else all(
        not counts['pending'] for counts in progress(manifest).values()
    )

    if args.progress:
        return 0 if complete or not args.require_complete else 1

    if not args.quiet:
        print_rankings(manifest)

    if not args.no_tables:
        spec_name = manifest['spec_name']
        docx_path = args.docx or os.path.join(args.tables_dir, f'{spec_name}_tuning.docx')
        csv_path = args.csv or os.path.join(args.tables_dir, f'{spec_name}_tuning.csv')
        written = write_tables(
            manifest, docx_path=docx_path, csv_path=csv_path, print_text=not args.quiet
        )
        for path in written:
            print(f"Wrote {path}")

    if args.require_complete and not complete:
        print("\nERROR: the sweep is incomplete and --require_complete was given.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
