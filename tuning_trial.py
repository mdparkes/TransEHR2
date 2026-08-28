#!/usr/bin/env python3
"""Look one trial up in a tuning manifest, for the SLURM array scripts.

A job array indexes trials by ``$SLURM_ARRAY_TASK_ID``. The mapping from an index to a config
path belongs with the manifest rather than in shell, so the array scripts call this instead of
globbing a directory -- a glob would silently reorder if a config were added, and the pretrain
and finetune arrays index different subsets of the same manifest.

Usage:
    python tuning_trial.py <manifest> --stage pretrain --count
    python tuning_trial.py <manifest> --stage pretrain --index 0 --field config
    python tuning_trial.py <manifest> --stage finetune --index 3 --field name
"""

import argparse
import sys

from hp_tuning.spec import finetune_trials, load_manifest, pretrain_trials


FIELDS = ('name', 'config', 'arm', 'hyperparameter', 'value')


def main(argv=None):
    """Print one field of one trial, or the number of trials in a stage.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        Process exit status. 1 if the index is out of range, so a mis-sized array fails the
        task rather than running the wrong trial.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('manifest', type=str)
    parser.add_argument('--stage', choices=['pretrain', 'finetune'], required=True,
                        help='Which subset of trials to index. Every trial needs a pretrain; '
                             'only those whose hyperparameter is selected downstream, plus '
                             'each arm\'s centre, need a finetune.')
    parser.add_argument('--arm', type=str, default=None,
                        help='Restrict to one encoding arm, so an array can run a single arm')
    parser.add_argument('--index', type=int, default=None,
                        help='Zero-based index into the stage, i.e. $SLURM_ARRAY_TASK_ID')
    parser.add_argument('--field', choices=FIELDS, default='config',
                        help='Which field to print. Default config.')
    parser.add_argument('--count', action='store_true',
                        help='Print the number of trials in the stage and exit. The largest '
                             'valid --array index is one less than this.')
    parser.add_argument('--list', action='store_true',
                        help='Print index and name for every trial in the stage')
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    chooser = pretrain_trials if args.stage == 'pretrain' else finetune_trials
    trials = chooser(manifest, arm=args.arm)

    if args.count:
        print(len(trials))
        return 0

    if args.list:
        for index, trial in enumerate(trials):
            print(f"{index}\t{trial['name']}")
        return 0

    if args.index is None:
        print("ERROR: one of --index, --count or --list is required", file=sys.stderr)
        return 2

    if not 0 <= args.index < len(trials):
        print(
            f"ERROR: index {args.index} is out of range for the {args.stage} stage, which has "
            f"{len(trials)} trials (valid indices 0-{len(trials) - 1}). Check the --array "
            f"range against `python {sys.argv[0]} {args.manifest} --stage {args.stage} "
            f"--count`.",
            file=sys.stderr
        )
        return 1

    value = trials[args.index][args.field]
    print('' if value is None else value)
    return 0


if __name__ == '__main__':
    sys.exit(main())
