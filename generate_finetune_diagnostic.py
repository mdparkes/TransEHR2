"""Expand a finetuning diagnostic: one pretrained encoder, a grid of finetuning schedules.

The grid crosses finetuning learning rates with decay half-lives. Every cell finetunes from
the same encoder, so the comparison is between finetuning settings and nothing else. Two
control cells sit alongside it at a fixed reference setting:

    random  the encoders are not loaded, so the run measures what the head reaches without
            anything pretraining produced. Read against the matching grid cell, the gap is
            pretraining's contribution.
    frozen  the encoders are loaded but excluded from the update, so the run measures the
            encoder as pretraining left it rather than as finetuning reshapes it.

Two variants are available. `history` includes prior admissions; `instay` restricts the record
to the current stay, which is the setting Xu et al. published on.

Expand with:
    python generate_finetune_diagnostic.py --variant instay

Then pretrain once, link the encoder into each cell, and finetune. The script prints the
commands.
"""

import argparse
import os

import yaml

BASE_CONFIG = 'TransEHR2/configs/experiments/tuning/phase2_base.yaml'
OUTPUT_DIR = 'TransEHR2/configs/experiments/ftdiag'

# The encoder every cell shares. Additive rather than RoPE because the failure appeared in
# both arms, so the cheaper one is enough, and a half-life of 40 anneals well inside a
# pretraining run whose best epoch lands between 50 and 110.
PRETRAIN_OVERRIDES = {
    'POSITION_ENCODING': 'additive',
    'PRETRAIN_LEARNING_RATE': 0.0006,
    'PRETRAIN_LR_HALF_LIFE': 40,
}

# `instay` drops prior admissions, which also moves the value stream's largest gap. The frozen
# ladder is spaced log-uniformly in period between P_MIN and P_MAX, with P_MAX ~ 63 * gap_max
# from the informative-band criterion 0.1 <= gap/lambda <= pi. With history the value stream
# spans 1 - 127,829 h, so P_MAX is 8.05e6; without it the span is the event stream's 1 - 48 h
# and P_MAX is 63 * 48 = 3024. Left at 8.05e6 for an in-stay run, half the ladder would sit at
# periods longer than any gap that occurs, phase-static and contributing a constant. P_MIN is
# 2 * gap_min and does not move: both variants have a smallest gap of one hour.
VARIANTS = {
    'history': {
        'prefix': 'ftdiag',
        'overrides': {'USE_HISTORICAL_RECORDS': True, 'VALUE_LADDER_P_MAX': 8.05e6},
    },
    'instay': {
        'prefix': 'ftdiag_instay',
        'overrides': {'USE_HISTORICAL_RECORDS': False, 'VALUE_LADDER_P_MAX': 3024.0},
    },
}

FINETUNE_LEARNING_RATES = [0.0002, 0.00005, 0.00001]

# None holds the rate constant, so a difference between it and the others is attributable to
# annealing rather than to the rate.
FINETUNE_HALF_LIVES = [None, 120, 160]

# The setting the controls run at, which must also be a cell of the grid so the three are read
# against each other at one rate and one schedule.
CONTROL_REFERENCE = {'FINETUNE_LEARNING_RATE': 0.00005, 'FINETUNE_LR_HALF_LIFE': None}
CONTROLS = {
    'ctl_random': {'FINETUNE_ENCODER_INIT': 'random'},
    'ctl_frozen': {'FINETUNE_FREEZE_ENCODER': True},
}

# Weights written by pretraining that a finetune reads back.
ENCODER_FILES = ('pretrained.pt', 'value_encoder.pt', 'event_encoder.pt')


def token(value):
    """Filename-safe token for a hyperparameter value."""
    if value is None:
        return 'flat'
    return f'{value:g}'.replace('.', 'p').replace('-', 'm').replace('+', '')


def cells(prefix):
    """The grid and the controls, as (name, overrides) pairs."""
    for rate in FINETUNE_LEARNING_RATES:
        for half_life in FINETUNE_HALF_LIVES:
            yield (f'{prefix}_lr{token(rate)}_hl{token(half_life)}',
                   {'FINETUNE_LEARNING_RATE': rate, 'FINETUNE_LR_HALF_LIFE': half_life})
    for name, overrides in CONTROLS.items():
        yield f'{prefix}_{name}', dict(CONTROL_REFERENCE, **overrides)


def write_config(base, name, overrides, variant_overrides, output_dir, header):
    """Write one experiment config, and return its path."""
    config = dict(base)
    config['EXPERIMENT_NAME'] = name
    config.update(PRETRAIN_OVERRIDES)
    config.update(variant_overrides)
    config.update(overrides)
    path = os.path.join(output_dir, f'{name}.yaml')
    with open(path, 'w') as f_out:
        f_out.write(header)
        yaml.safe_dump(config, f_out, default_flow_style=False, sort_keys=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default='history', choices=sorted(VARIANTS),
                        help='Record scope the whole diagnostic runs under')
    parser.add_argument('--base', default=BASE_CONFIG, help='Config the cells inherit from')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Where to write the configs')
    parser.add_argument('--fold', default='fold0', help='Fold the diagnostic runs on')
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    prefix = variant['prefix']
    pretrain_name = f'{prefix}_pretrain'

    with open(args.base) as f_in:
        base = yaml.safe_load(f_in)
    base.pop('HYPERPARAMETERS_TO_TUNE', None)
    os.makedirs(args.output_dir, exist_ok=True)

    header = (f'# Generated by generate_finetune_diagnostic.py --variant {args.variant}\n'
              f'# from {args.base}. Do not edit by hand -- regenerate instead.\n\n')

    pretrain_path = write_config(base, pretrain_name, {}, variant['overrides'],
                                 args.output_dir, header)
    cell_paths = [(name, write_config(base, name, overrides, variant['overrides'],
                                      args.output_dir, header))
                  for name, overrides in cells(prefix)]

    model_dir = base['MODEL_DIR']
    print(f'Variant:      {args.variant}  '
          f'(USE_HISTORICAL_RECORDS={variant["overrides"]["USE_HISTORICAL_RECORDS"]})')
    print(f'Encoder:      {pretrain_name}')
    print(f'Cells:        {len(cell_paths)}  '
          f'({len(FINETUNE_LEARNING_RATES)} rates x {len(FINETUNE_HALF_LIVES)} half-lives '
          f'+ {len(CONTROLS)} controls)\n')
    for name, path in cell_paths:
        print(f'  {name:<40} {path}')

    # slurm_run_experiment.sh indexes FOLDS by the array id and defaults to the five
    # manuscript folds, so a single-fold run has to name the fold and take one array task.
    launch = f'FOLDS="{args.fold}" sbatch --array=0-0 SLURM/slurm_run_experiment.sh'
    print('\n1. Pretrain once:')
    print(f'   TASKS=none {launch} {pretrain_path}')
    print('\n2. Link that encoder into every cell, so each one skips pretraining and they all')
    print('   finetune from identical weights:')
    print(f'   src="{model_dir}/{pretrain_name}/{args.fold}/pretrained"')
    print(f'   for cell in {" ".join(name for name, _ in cell_paths)}; do')
    print(f'     dst="{model_dir}/$cell/{args.fold}/pretrained"; mkdir -p "$dst"')
    print(f'     for f in {" ".join(ENCODER_FILES)}; do ln -sf "$src/$f" "$dst/$f"; done')
    print('   done')
    print('\n3. Finetune every cell:')
    print(f'   for cfg in {args.output_dir}/{prefix}_lr*.yaml '
          f'{args.output_dir}/{prefix}_ctl_*.yaml; do')
    print(f'     TASKS=mortality {launch} "$cfg"')
    print('   done')


if __name__ == '__main__':
    main()
