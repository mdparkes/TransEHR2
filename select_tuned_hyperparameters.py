#!/usr/bin/env python3
"""Pick the winning hyperparameter values from a finished sweep and assemble a config.

This is the step between one phase and the next: it reads the rankings, records the decision
together with the numbers it was made from, and writes a complete experiment config with every
tuned hyperparameter set to its winner. That config is what Phase 3 sweeps history lengths
against and what Phase 4 runs across the manuscript folds.

Usage:
    # Decide, and write the assembled config for the winning arm
    python select_tuned_hyperparameters.py <manifest> \\
        --output TransEHR2/configs/experiments/experiment10_tuned.yaml \\
        --experiment_name experiment10_tuned

    # Force a particular arm rather than taking the head-to-head winner
    python select_tuned_hyperparameters.py <manifest> --arm additive --output <path>

    # See the decision without writing anything
    python select_tuned_hyperparameters.py <manifest> --dry_run

An additive sweep returns independent winners whose *combination* is never run. That is a
known and accepted gap -- the revision plan says so, and Phase 4 validates the assembled
configuration on the manuscript folds. The gap is recorded in the selection YAML rather than
papered over, so it stays visible when the methods section gets written.
"""

import argparse
import os
import sys

import yaml

from hp_tuning.reporting import format_grid_value, format_metric
from hp_tuning.results import compare_arms, rank_hyperparameter
from hp_tuning.spec import load_manifest


def choose_arm(manifest, requested):
    """Decide which encoding arm to carry forward.

    Args:
        manifest: A loaded manifest.
        requested: An arm name, or None to take the head-to-head winner.

    Returns:
        A tuple of (arm name, the comparison dict or None if the arm was named outright).

    Raises:
        ValueError: If the requested arm is not in the sweep, or if no arm produced a usable
            result at its centre.
    """
    if requested is not None:
        if requested not in manifest['arms']:
            raise ValueError(
                f"Arm {requested!r} is not in this sweep. Available: {manifest['arms']}"
            )
        return requested, None

    if len(manifest['arms']) == 1:
        return manifest['arms'][0], None

    comparison = compare_arms(manifest)
    if comparison['best'] is None:
        raise ValueError(
            "No arm produced a usable result at its centre, so the encoding cannot be chosen. "
            "Run report_tuning_results.py to see why, or name an arm with --arm."
        )
    return comparison['best'].trial['arm'], comparison


def main(argv=None):
    """Select the tuned hyperparameters and write the assembled config.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('manifest', type=str, help='Manifest written by generate_tuning_configs.py')
    parser.add_argument('--arm', type=str, default=None,
                        help='Encoding arm to carry forward. Defaults to the head-to-head '
                             'winner at the shared centre.')
    parser.add_argument('--output', type=str, default=None,
                        help='Where to write the assembled experiment config')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='EXPERIMENT_NAME for the assembled config. Defaults to the '
                             'output filename without its extension.')
    parser.add_argument('--selection_yaml', type=str, default=None,
                        help='Where to record the decision and the numbers behind it. '
                             'Defaults to <manifest dir>/<spec>_selection.yaml')
    parser.add_argument('--allow_incomplete', action='store_true',
                        help='Select even though some grid values produced no usable result. '
                             'The winner is then the best of what finished, which is a '
                             'different and weaker claim.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the decision without writing anything')
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if manifest.get('design') == 'factorial':
        # A factorial's winner is a cell that was actually run, so there is nothing to
        # assemble from coordinate winners and the per-hyperparameter ranking this script
        # performs would discard the interaction the design exists to measure.
        print(f"ERROR: {args.manifest} is a factorial sweep. Selection over a cross product "
              f"ranks whole cells, not coordinates.\n"
              f"       Use select_tuned_cell.py instead.", file=sys.stderr)
        return 1
    arm, comparison = choose_arm(manifest, args.arm)

    print(f"Sweep:    {manifest['spec_name']}")
    print(f"Fold:     {manifest['fold']}")
    if comparison is not None:
        print(f"Arm:      {arm}  (head-to-head winner on "
              f"{comparison['metric']} at the shared centre)")
        for result in comparison['results']:
            shown = format_metric(result.value) if result.is_usable else f'[{result.status}]'
            print(f"            {str(result.grid_value):<10} {shown}")
    else:
        print(f"Arm:      {arm}  (named on the command line)")
    print()

    rankings = [rank_hyperparameter(manifest, arm, name) for name in manifest['grid']]
    incomplete = [r['hyperparameter'] for r in rankings if not r['complete']]
    undecidable = [r['hyperparameter'] for r in rankings if r['best'] is None]

    print(f"Selected hyperparameters, {arm} arm")
    print("-" * 70)
    selected = {}
    for ranking in rankings:
        name = ranking['hyperparameter']
        if ranking['best'] is None:
            print(f"  {name:<32} UNDECIDED -- no usable result for any value")
            continue
        selected[name] = ranking['best'].grid_value
        flag = '  (incomplete grid)' if not ranking['complete'] else ''
        print(f"  {name:<32} {format_grid_value(ranking['best'].grid_value):<10} "
              f"{ranking['metric']} = {format_metric(ranking['best'].value)}{flag}")
    print()

    if undecidable:
        print(f"ERROR: no value could be selected for {undecidable}. "
              f"Run report_tuning_results.py to see why.", file=sys.stderr)
        return 1

    if incomplete and not args.allow_incomplete:
        print(f"ERROR: the grid is incomplete for {incomplete}, so the winners above are the "
              f"best of the trials that finished rather than the best of the grid.\n"
              f"       Re-run the missing trials, or pass --allow_incomplete to accept that.",
              file=sys.stderr)
        return 1

    arm_overrides = {}
    for trial in manifest['trials']:
        if trial['arm'] == arm and trial['is_centre']:
            # The arm's defining settings are whatever its centre carries that is not a tuned
            # hyperparameter -- POSITION_ENCODING, and anything else the spec's ARMS block set.
            with open(trial['config'], 'r') as f_in:
                centre_config = yaml.safe_load(f_in)
            arm_overrides = {
                key: value for key, value in centre_config.items()
                if key not in manifest['grid'] and key != 'EXPERIMENT_NAME'
            }
            break
    if not arm_overrides:
        print(f"ERROR: could not read the centre config for arm {arm!r}; the assembled config "
              f"would be missing everything the arm sets.", file=sys.stderr)
        return 1

    assembled = dict(arm_overrides)
    assembled.update(selected)

    if args.output:
        experiment_name = args.experiment_name or os.path.splitext(
            os.path.basename(args.output)
        )[0]
        assembled['EXPERIMENT_NAME'] = experiment_name
    elif args.experiment_name:
        assembled['EXPERIMENT_NAME'] = args.experiment_name

    decision = {
        'spec_name': manifest['spec_name'],
        'manifest': os.path.abspath(args.manifest),
        'fold': manifest['fold'],
        'arm': arm,
        'arm_chosen_by': 'head-to-head at the shared centre' if comparison else 'command line',
        'selected': {
            ranking['hyperparameter']: {
                'value': ranking['best'].grid_value,
                'criterion': ranking['criterion'],
                'metric': ranking['metric'],
                'direction': ranking['direction'],
                'metric_value': ranking['best'].value,
                'trial': ranking['best'].name,
                'grid': manifest['grid'][ranking['hyperparameter']]['values'],
                'complete': ranking['complete'],
                'measured': {
                    format_grid_value(r.grid_value): (r.value if r.is_usable else r.status)
                    for r in ranking['results']
                },
            }
            for ranking in rankings if ranking['best'] is not None
        },
        'caveats': [
            'Additive sweep: each hyperparameter was tuned against the others at their '
            'defaults, so this combination of winners was never run as a configuration. '
            'Phase 4 validates it on the manuscript folds.',
            'Hyperparameters were selected under the tuning epoch budget and are applied to '
            'much longer final runs. Learning rate decay is the one most sensitive to that '
            'gap and the one most worth a line in the methods section.',
            f'Selected on {manifest["fold"]} alone, which is the fold held out of the '
            f'manuscript results for exactly this purpose.',
        ],
    }
    if comparison is not None:
        decision['arm_comparison'] = {
            'metric': comparison['metric'],
            'direction': comparison['direction'],
            'measured': {
                str(r.grid_value): (r.value if r.is_usable else r.status)
                for r in comparison['results']
            },
            'caveat': 'The arms were compared at their shared centre rather than at each '
                      'arm\'s assembled optimum. A confirmation stage of four runs would '
                      'close that gap.',
        }
    if incomplete:
        decision['incomplete_grids'] = incomplete

    if args.dry_run:
        print("Dry run: nothing written. The assembled config would be:\n")
        print(yaml.dump(assembled, default_flow_style=False, sort_keys=False))
        return 0

    selection_path = args.selection_yaml or os.path.join(
        os.path.dirname(os.path.abspath(args.manifest)),
        f"{manifest['spec_name']}_selection.yaml"
    )
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    with open(selection_path, 'w') as f_out:
        yaml.dump(decision, f_out, default_flow_style=False, sort_keys=False)
    print(f"Wrote the decision and the numbers behind it to {selection_path}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        header = (
            f"# Assembled by select_tuned_hyperparameters.py from {selection_path}\n"
            f"# Sweep {manifest['spec_name']}, arm {arm}, selected on {manifest['fold']}.\n"
            f"#\n"
            f"# Each tuned value won its own coordinate against the others at their defaults.\n"
            f"# This combination was not itself run during tuning; that is the known gap in an\n"
            f"# additive sweep, and it is what the manuscript folds validate.\n"
            f"\n"
        )
        with open(args.output, 'w') as f_out:
            f_out.write(header)
            yaml.dump(assembled, f_out, default_flow_style=False, sort_keys=False)
        print(f"Wrote the assembled config to {args.output}")
    else:
        print("\nNo --output given, so no assembled config was written.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
