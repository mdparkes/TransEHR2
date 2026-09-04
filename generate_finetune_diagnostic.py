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


def parse_values(text, cast):
    """Parse a comma-separated grid argument.

    Args:
        text: The argument, e.g. "5e-5,2.2e-5,1e-5". "flat" and "none" mean no decay.
        cast: Callable applied to each non-flat entry.

    Returns:
        list: The parsed values, in the order given. The first is the default.
    """
    values = []
    for piece in text.split(','):
        piece = piece.strip()
        values.append(None if piece.lower() in ('flat', 'none') else cast(piece))
    return values


def token(value):
    """Filename-safe token for a hyperparameter value."""
    if value is None:
        return 'flat'
    return f'{value:g}'.replace('.', 'p').replace('-', 'm').replace('+', '')


def cells(prefix, rates, half_lives, controls=True):
    """The grid, and the controls if they are wanted, as (name, overrides) pairs.

    Args:
        prefix: Name prefix for every cell.
        rates: Finetuning learning rates.
        half_lives: Finetuning decay half-lives; None holds the rate constant.
        controls: Whether to emit the random-encoder and frozen-encoder cells.

    Yields:
        Tuples of (experiment name, config overrides).
    """
    for rate in rates:
        for half_life in half_lives:
            yield (f'{prefix}_lr{token(rate)}_hl{token(half_life)}',
                   {'FINETUNE_LEARNING_RATE': rate, 'FINETUNE_LR_HALF_LIFE': half_life})
    if not controls:
        return
    for name, overrides in CONTROLS.items():
        yield f'{prefix}_{name}', dict(CONTROL_REFERENCE, **overrides)


def write_config(base, name, overrides, upstream, output_dir, header):
    """Write one experiment config, and return its path.

    Args:
        base: The config every cell inherits.
        name: EXPERIMENT_NAME for this cell.
        overrides: The cell's own settings.
        upstream: Pretraining and record-scope settings to force. Empty when the base config
            is authoritative, which is the case when an existing encoder is being reused.
        output_dir: Where to write.
        header: Provenance comment for the top of the file.
    """
    config = dict(base)
    config['EXPERIMENT_NAME'] = name
    config.update(upstream)
    config.update(overrides)
    path = os.path.join(output_dir, f'{name}.yaml')
    with open(path, 'w') as f_out:
        f_out.write(header)
        yaml.safe_dump(config, f_out, default_flow_style=False, sort_keys=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default=None, choices=sorted(VARIANTS),
                        help='Record scope to force on every cell, and pretrain an encoder '
                             'for. Mutually exclusive with --encoder.')
    parser.add_argument('--encoder', default=None,
                        help='EXPERIMENT_NAME of an existing pretrain whose weights every '
                             'cell links. No pretrain config is written and the base config '
                             'is taken as authoritative, so nothing upstream is overridden.')
    parser.add_argument('--prefix', default=None,
                        help='Name prefix for the cells. Defaults to the variant\'s prefix, '
                             'or to the base config filename when --encoder is given.')
    parser.add_argument('--base', default=BASE_CONFIG, help='Config the cells inherit from')
    parser.add_argument('--rates', default='0.0002,0.00005,0.00001',
                        help='Finetuning learning rates, comma separated. First is default.')
    parser.add_argument('--half_lives', default='flat,120,160',
                        help='Finetuning decay half-lives in epochs, comma separated. '
                             '"flat" holds the rate constant.')
    parser.add_argument('--no_controls', action='store_true',
                        help='Omit the random-encoder and frozen-encoder cells')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Where to write the configs')
    parser.add_argument('--fold', default='fold0', help='Fold the cells run on')
    args = parser.parse_args()

    if args.encoder and args.variant:
        parser.error('--encoder reuses a pretrain whose record scope is already fixed, so '
                     '--variant would silently disagree with it. Pass one or the other.')

    rates = parse_values(args.rates, float)
    half_lives = parse_values(args.half_lives, float)

    if args.encoder:
        upstream = {}
        prefix = args.prefix or os.path.splitext(os.path.basename(args.base))[0]
        encoder_name = args.encoder
    else:
        variant = VARIANTS[args.variant or 'history']
        upstream = dict(PRETRAIN_OVERRIDES, **variant['overrides'])
        prefix = args.prefix or variant['prefix']
        encoder_name = f'{prefix}_pretrain'

    with open(args.base) as f_in:
        base = yaml.safe_load(f_in)
    base.pop('HYPERPARAMETERS_TO_TUNE', None)
    os.makedirs(args.output_dir, exist_ok=True)

    header = (f'# Generated by generate_finetune_diagnostic.py from {args.base}\n'
              f'# Do not edit by hand -- regenerate instead.\n\n')

    pretrain_path = None
    if not args.encoder:
        pretrain_path = write_config(base, encoder_name, {}, upstream, args.output_dir, header)
    cell_paths = [(name, write_config(base, name, overrides, upstream, args.output_dir, header))
                  for name, overrides in cells(prefix, rates, half_lives,
                                               controls=not args.no_controls)]

    model_dir = base['MODEL_DIR']
    print(f'Base config:  {args.base}')
    print(f'Encoder:      {encoder_name}'
          f'{"  (existing, linked)" if args.encoder else "  (pretrained by step 1)"}')
    print(f'Cells:        {len(cell_paths)}  '
          f'({len(rates)} rates x {len(half_lives)} half-lives'
          f'{"" if args.no_controls else f" + {len(CONTROLS)} controls"})\n')
    for name, path in cell_paths:
        print(f'  {name:<44} {path}')

    # slurm_run_experiment.sh indexes FOLDS by the array id and defaults to the five
    # manuscript folds, so a single-fold run has to name the fold and take one array task.
    launch = f'FOLDS="{args.fold}" sbatch --array=0-0 SLURM/slurm_run_experiment.sh'
    step = 1
    if pretrain_path is not None:
        print(f'\n{step}. Pretrain once:')
        print(f'   TASKS=none {launch} {pretrain_path}')
        step += 1
    print(f'\n{step}. Link that encoder into every cell, so each one skips pretraining and')
    print('   they all finetune from identical weights:')
    print(f'   src="{model_dir}/{encoder_name}/{args.fold}/pretrained"')
    print(f'   for cfg in {args.output_dir}/{prefix}_*.yaml; do')
    print('     cell=$(basename "$cfg" .yaml)')
    # The glob also matches the encoder's own config, and linking its weights over themselves
    # would replace the files with symlinks to themselves.
    print(f'     [ "$cell" = "{encoder_name}" ] && continue')
    print(f'     dst="{model_dir}/$cell/{args.fold}/pretrained"; mkdir -p "$dst"')
    print(f'     for f in {" ".join(ENCODER_FILES)}; do ln -sf "$src/$f" "$dst/$f"; done')
    print('   done')
    step += 1
    print(f'\n{step}. Finetune every cell:')
    print(f'   for cfg in {args.output_dir}/{prefix}_lr*.yaml'
          f'{"" if args.no_controls else f" {args.output_dir}/{prefix}_ctl_*.yaml"}; do')
    print(f'     TASKS=mortality {launch} "$cfg"')
    print('   done')


if __name__ == '__main__':
    main()
