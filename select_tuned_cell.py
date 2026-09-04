"""Select the winning cell of a factorial sweep and write it as an experiment config.

A factorial is run where the hyperparameters interact, which is the case in which the best
combination is not the combination of individual bests. Selection therefore ranks whole cells,
and the winner is a configuration that was actually run rather than one assembled from
coordinate winners -- so the config written here is the winning trial's own, renamed.

    python select_tuned_cell.py <manifest>                       # report every arm
    python select_tuned_cell.py <manifest> --arm rope --output <path>

Use select_tuned_hyperparameters.py for an additive sweep.
"""

import argparse
import os
import sys

import yaml

from hp_tuning.reporting import format_grid_value, format_metric
from hp_tuning.results import rank_cells
from hp_tuning.spec import load_manifest

CAVEATS = [
    'Factorial sweep: the selected cell was run as a configuration, so no combination of '
    'separately chosen winners is being assumed.',
    'Hyperparameters were selected under the tuning epoch budget and are applied to much '
    'longer final runs. The learning rate half-life is the one most sensitive to that gap '
    'and the one most worth a line in the methods section.',
]


def describe_cell(cell):
    """Render a cell's assignment on one line.

    Args:
        cell: A mapping of hyperparameter name to value.

    Returns:
        str: The assignment, space separated.
    """
    return '  '.join(f'{name}={format_grid_value(value)}' for name, value in cell.items())


def print_arm(ranking):
    """Print one arm's cells, best first.

    Args:
        ranking: A ranking dict as returned by :func:`hp_tuning.results.rank_cells`.
    """
    direction = 'lowest' if ranking['direction'] == 'min' else 'highest'
    print(f"  {ranking['arm']}: {direction} {ranking['metric']} "
          f"on {ranking['criterion']}")
    usable = [r for r in ranking['results'] if r.is_usable]
    pending = [r for r in ranking['results'] if not r.is_usable]
    ordered = sorted(usable, key=lambda r: r.value, reverse=ranking['direction'] == 'max')
    for result in ordered + pending:
        marker = '  <-- selected' if result is ranking['best'] else ''
        shown = format_metric(result.value) if result.is_usable else f'[{result.status}]'
        print(f"    {describe_cell(result.grid_value):<60} {shown:>12}{marker}")
    if not ranking['complete']:
        print("    NOTE incomplete: the winner is the best of the cells that finished, which")
        print("         is a weaker claim than the best of the grid.")
    print()


def main(argv=None):
    """Report the winning cell per arm, and write one arm's config on request.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('manifest', type=str,
                        help='Manifest written by generate_tuning_configs.py')
    parser.add_argument('--arm', type=str, default=None,
                        help='Arm to write a config for. Without it, every arm is reported '
                             'and nothing is written. A factorial phase that carries both '
                             'arms forward is run once per arm.')
    parser.add_argument('--output', type=str, default=None,
                        help='Where to write the selected experiment config')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='EXPERIMENT_NAME for the written config. Defaults to the output '
                             'filename without its extension.')
    parser.add_argument('--selection_yaml', type=str, default=None,
                        help='Where to record the decision and the numbers behind it. '
                             'Defaults to <manifest dir>/<spec>_<arm>_cell.yaml')
    parser.add_argument('--allow_incomplete', action='store_true',
                        help='Select even though some cells produced no usable result. The '
                             'winner is then the best of what finished, which is a different '
                             'and weaker claim.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the decision without writing anything')
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if manifest.get('design') != 'factorial':
        print(f"ERROR: {args.manifest} is a {manifest.get('design', 'additive')!r} sweep. "
              f"Use select_tuned_hyperparameters.py.", file=sys.stderr)
        return 1

    print(f"Sweep:    {manifest['spec_name']}")
    print(f"Fold:     {manifest['fold']}")
    print(f"Design:   factorial, {len(manifest['grid'])} hyperparameters\n")

    if args.arm is None:
        for arm in manifest['arms']:
            print_arm(rank_cells(manifest, arm))
        print("Pass --arm and --output to write one arm's config.")
        return 0

    if args.arm not in manifest['arms']:
        print(f"ERROR: arm {args.arm!r} is not in this sweep. Available: {manifest['arms']}",
              file=sys.stderr)
        return 1

    ranking = rank_cells(manifest, args.arm)
    print_arm(ranking)

    if ranking['best'] is None:
        print("ERROR: no cell produced a usable result. Run report_tuning_results.py to see "
              "why.", file=sys.stderr)
        return 1
    if not ranking['complete'] and not args.allow_incomplete:
        print("ERROR: some cells have no usable result, so the winner above is the best of "
              "the cells that finished rather than the best of the grid.\n"
              "       Re-run the missing cells, or pass --allow_incomplete to accept that.",
              file=sys.stderr)
        return 1

    winner = ranking['best']
    with open(winner.trial['config'], 'r') as f_in:
        selected_config = {
            key: value for key, value in yaml.safe_load(f_in).items()
            if key != 'EXPERIMENT_NAME'
        }

    if args.output:
        selected_config['EXPERIMENT_NAME'] = args.experiment_name or os.path.splitext(
            os.path.basename(args.output)
        )[0]
    elif args.experiment_name:
        selected_config['EXPERIMENT_NAME'] = args.experiment_name

    decision = {
        'spec_name': manifest['spec_name'],
        'manifest': os.path.abspath(args.manifest),
        'fold': manifest['fold'],
        'arm': args.arm,
        'design': 'factorial',
        'criterion': ranking['criterion'],
        'metric': ranking['metric'],
        'direction': ranking['direction'],
        'selected': {
            'cell': dict(winner.grid_value),
            'trial': winner.name,
            'metric_value': winner.value,
            'complete': ranking['complete'],
        },
        'measured': {
            describe_cell(r.grid_value): (r.value if r.is_usable else r.status)
            for r in ranking['results']
        },
        'caveats': CAVEATS + [
            f'Selected on {manifest["fold"]} alone, which is the fold held out of the '
            f'manuscript results for exactly this purpose.',
        ],
    }
    if not ranking['complete']:
        decision['incomplete'] = True

    if args.dry_run:
        print("Dry run: nothing written. The selected config would be:\n")
        print(yaml.dump(selected_config, default_flow_style=False, sort_keys=False))
        return 0

    selection_path = args.selection_yaml or os.path.join(
        os.path.dirname(os.path.abspath(args.manifest)),
        f"{manifest['spec_name']}_{args.arm}_cell.yaml"
    )
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    with open(selection_path, 'w') as f_out:
        yaml.dump(decision, f_out, default_flow_style=False, sort_keys=False)
    print(f"Wrote the decision and the numbers behind it to {selection_path}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        header = (
            f"# Selected by select_tuned_cell.py from {selection_path}\n"
            f"# The winning cell of {manifest['spec_name']}, {args.arm} arm, as run.\n\n"
        )
        with open(args.output, 'w') as f_out:
            f_out.write(header)
            yaml.dump(selected_config, f_out, default_flow_style=False, sort_keys=False)
        print(f"Wrote the selected config to {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
