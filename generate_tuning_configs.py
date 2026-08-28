#!/usr/bin/env python3
"""Expand a hyperparameter tuning spec into one experiment config per trial.

The sweep is additive: a shared all-defaults centre per encoding arm, plus one trial for each
non-default value of each hyperparameter. Five hyperparameters at three values each is eleven
configurations per arm, and each becomes a standalone config that a single-GPU
``run_experiment.py`` job runs on its own.

Usage:
    python generate_tuning_configs.py TransEHR2/configs/experiments/tuning/phase2_spec.yaml

    # See what it would write without writing anything
    python generate_tuning_configs.py <spec> --dry_run

    # Print the sbatch commands the manifest implies
    python generate_tuning_configs.py <spec> --dry_run --show_commands

Regenerating over existing configs needs --overwrite, because a config file is the record of
what a finished run actually ran. Rewriting one silently re-describes results already on disk.
"""

import argparse
import sys

from hp_tuning.spec import (SELECTION_CRITERIA, expand_trials, load_spec, write_manifest,
                            write_trial_configs)


def main(argv=None):
    """Generate the trial configs and the manifest.

    Args:
        argv: Command-line arguments, or None to read sys.argv.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('spec', type=str, help='Tuning spec YAML')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace trial configs that already exist')
    parser.add_argument('--dry_run', action='store_true',
                        help='List the trials without writing anything')
    parser.add_argument('--show_commands', action='store_true',
                        help='Print the sbatch commands for the pretrain and finetune stages')
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    trials = expand_trials(spec)

    n_arms = len(spec['ARMS'])
    n_finetune = sum(1 for trial in trials if trial['needs_finetune'])

    print(f"Spec:        {spec['SPEC_PATH']}")
    print(f"Base config: {spec['BASE_CONFIG']}")
    print(f"Output dir:  {spec['OUTPUT_DIR']}")
    print(f"Manifest:    {spec['MANIFEST']}")
    print(f"Fold:        {spec['FOLD']}")
    print(f"Arms:        {', '.join(spec['ARMS'])}")
    print()
    print("Grid, first value of each is the default and is carried by the shared centre:")
    for name, entry in spec['GRID'].items():
        criterion = SELECTION_CRITERIA[entry['select_on']]
        print(f"  {name:<32} {entry['values']}")
        print(f"  {'':<32} selected on {entry['select_on']} "
              f"({criterion['direction']} {criterion['metric']})")
    print()
    print(f"{len(trials)} trials: {len(trials) // n_arms} per arm across {n_arms} arms.")
    print(f"{len(trials)} pretraining jobs, {n_finetune} finetuning jobs.")
    print()

    for trial in trials:
        if trial['is_centre']:
            described = 'centre (all defaults)'
        else:
            described = f"{trial['hyperparameter']} = {trial['value']!r}"
        stages = 'pretrain + finetune' if trial['needs_finetune'] else 'pretrain'
        print(f"  [{trial['arm']:<8}] {trial['name']:<52} {described:<44} {stages}")

    if args.dry_run:
        print("\nDry run: nothing written.")
    else:
        trials = write_trial_configs(spec, trials, overwrite=args.overwrite)
        manifest_path = write_manifest(spec, trials)
        print(f"\nWrote {len(trials)} configs to {spec['OUTPUT_DIR']}")
        print(f"Wrote manifest to {manifest_path}")

    if args.show_commands or args.dry_run:
        manifest_path = spec['MANIFEST']
        print("\nTo run the sweep, in order:")
        print(f"  sbatch --array=0-{len(trials) - 1} "
              f"SLURM/slurm_tune_pretrain.sh {manifest_path}")
        print(f"  sbatch --array=0-{n_finetune - 1} "
              f"SLURM/slurm_tune_finetune.sh {manifest_path}")
        print(f"  sbatch SLURM/slurm_report_tuning.sh {manifest_path}")
        print()
        print("The finetune array depends on the pretrain array: every finetune loads the "
              "encoder weights its own pretrain wrote. Chain them with")
        print(f"  --dependency=afterok:<pretrain job id>")
        print("or wait for the pretrain array to drain before submitting the second.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
