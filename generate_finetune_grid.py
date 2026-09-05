"""Expand a grid of finetuning configurations over one shared, already-pretrained encoder.

Every cell links the same weights, so the only thing separating them is the finetuning
settings. Two shapes:

    grid          --rates and --half_lives, crossed.
    seed repeats  --seeds, one run per seed at a single configuration. Give two groups the
                  same seeds and their comparison becomes within-seed.

Two control cells are added unless --no_controls: one finetunes from a fresh initialization
instead of the linked weights, the other freezes the encoder. Read against the matching grid
cell they separate what pretraining contributed from what finetuning the encoder contributed.

    python generate_finetune_grid.py --base <selected>.yaml --encoder <trial> \
        --prefix phase2b_additive --output_dir <dir> \
        --rates 5e-5,2.2e-5,1e-5 --half_lives 160,60,20 --no_controls

The script prints the commands to link and to submit.
"""

import argparse
import os

import yaml

# Weights pretraining writes that a finetune reads back.
ENCODER_FILES = ('pretrained.pt', 'value_encoder.pt', 'event_encoder.pt')

# The controls run at one grid cell, so the three are read against each other at one rate and
# one schedule.
CONTROL_REFERENCE = {'FINETUNE_LEARNING_RATE': 0.00005, 'FINETUNE_LR_HALF_LIFE': None}
CONTROLS = {
    'ctl_random': {'FINETUNE_ENCODER_INIT': 'random'},
    'ctl_frozen': {'FINETUNE_FREEZE_ENCODER': True},
}


def parse_values(text, cast):
    """Parse a comma-separated grid argument. 'flat' and 'none' mean no decay.

    Args:
        text: The argument, e.g. "5e-5,2.2e-5,1e-5".
        cast: Callable applied to each non-flat entry.

    Returns:
        list: The parsed values. The first is the default.
    """
    return [None if piece.strip().lower() in ('flat', 'none') else cast(piece.strip())
            for piece in text.split(',')]


def token(value):
    """Filename-safe token for a hyperparameter value."""
    if value is None:
        return 'flat'
    return f'{value:g}'.replace('.', 'p').replace('-', 'm').replace('+', '')


def grid_cells(prefix, rates, half_lives, controls):
    """The rate-by-half-life grid, and the controls if wanted.

    Args:
        prefix: Name prefix for every cell.
        rates: Finetuning learning rates.
        half_lives: Decay half-lives in epochs; None holds the rate constant.
        controls: Whether to emit the two control cells.

    Yields:
        Tuples of (experiment name, config overrides).
    """
    for rate in rates:
        for half_life in half_lives:
            yield (f'{prefix}_lr{token(rate)}_hl{token(half_life)}',
                   {'FINETUNE_LEARNING_RATE': rate, 'FINETUNE_LR_HALF_LIFE': half_life})
    if controls:
        for name, overrides in CONTROLS.items():
            yield f'{prefix}_{name}', dict(CONTROL_REFERENCE, **overrides)


def seed_cells(prefix, rate, half_life, seeds):
    """One cell per seed at a single configuration.

    Args:
        prefix: Name prefix for every cell.
        rate: The finetuning learning rate every repeat uses.
        half_life: The decay half-life every repeat uses.
        seeds: One repeat each.

    Yields:
        Tuples of (experiment name, config overrides).
    """
    for seed in seeds:
        yield (f'{prefix}_seed{seed}',
               {'FINETUNE_LEARNING_RATE': rate,
                'FINETUNE_LR_HALF_LIFE': half_life,
                'FINETUNE_SEED': seed})


def write_config(base, name, overrides, output_dir, header):
    """Write one experiment config and return its path.

    Args:
        base: The config every cell inherits, which is authoritative for everything
            upstream of finetuning.
        name: EXPERIMENT_NAME for this cell.
        overrides: The cell's own settings.
        output_dir: Where to write.
        header: Provenance comment for the top of the file.

    Returns:
        str: The path written.
    """
    config = dict(base)
    config['EXPERIMENT_NAME'] = name
    config.update(overrides)
    path = os.path.join(output_dir, f'{name}.yaml')
    with open(path, 'w') as f_out:
        f_out.write(header)
        yaml.safe_dump(config, f_out, default_flow_style=False, sort_keys=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True,
                        help='Config every cell inherits, from the phase that selected the '
                             'pretraining settings')
    parser.add_argument('--encoder', required=True,
                        help='EXPERIMENT_NAME of the pretrain whose weights every cell links')
    parser.add_argument('--prefix', required=True, help='Name prefix for the cells')
    parser.add_argument('--output_dir', required=True, help='Where to write the configs')
    parser.add_argument('--rates', default='0.00005,0.000022,0.00001',
                        help='Finetuning learning rates, comma separated')
    parser.add_argument('--half_lives', default='160,60,20',
                        help="Decay half-lives in epochs, comma separated. 'flat' holds the "
                             'rate constant.')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seeds. Replaces the grid with one repeat per '
                             'seed, so --rates and --half_lives must each name one value.')
    parser.add_argument('--no_controls', action='store_true',
                        help='Omit the random-encoder and frozen-encoder cells')
    parser.add_argument('--fold', default='fold0', help='Fold the cells run on')
    args = parser.parse_args()

    rates = parse_values(args.rates, float)
    half_lives = parse_values(args.half_lives, float)
    seeds = parse_values(args.seeds, int) if args.seeds else None
    if seeds is not None and (len(rates) != 1 or len(half_lives) != 1):
        parser.error('--seeds repeats one configuration, so --rates and --half_lives must '
                     'each name a single value.')

    with open(args.base) as f_in:
        base = yaml.safe_load(f_in)
    base.pop('HYPERPARAMETERS_TO_TUNE', None)
    os.makedirs(args.output_dir, exist_ok=True)

    header = (f'# Generated by generate_finetune_grid.py from {args.base}\n'
              f'# Do not edit by hand -- regenerate instead.\n\n')
    plan = (seed_cells(args.prefix, rates[0], half_lives[0], seeds) if seeds is not None
            else grid_cells(args.prefix, rates, half_lives, not args.no_controls))
    cell_paths = [(name, write_config(base, name, overrides, args.output_dir, header))
                  for name, overrides in plan]

    model_dir = base['MODEL_DIR']
    print(f'Base config:  {args.base}')
    print(f'Encoder:      {args.encoder}')
    if seeds is not None:
        print(f'Cells:        {len(cell_paths)}  (one per seed at lr {rates[0]:g}, '
              f'half-life {half_lives[0]:g})\n')
    else:
        print(f'Cells:        {len(cell_paths)}  ({len(rates)} rates x '
              f'{len(half_lives)} half-lives'
              f'{"" if args.no_controls else f" + {len(CONTROLS)} controls"})\n')
    for name, path in cell_paths:
        print(f'  {name:<44} {path}')

    # slurm_run_experiment.sh indexes FOLDS by the array id and defaults to the five
    # manuscript folds, so a single-fold run has to name the fold and take one array task.
    launch = f'FOLDS="{args.fold}" sbatch --array=0-0 SLURM/slurm_run_experiment.sh'
    suffix = 'seed*' if seeds is not None else '*'
    print('\n1. Link the encoder into every cell, so each skips pretraining and they all')
    print('   finetune from identical weights:')
    print(f'   src="{model_dir}/{args.encoder}/{args.fold}/pretrained"')
    print(f'   for cfg in {args.output_dir}/{args.prefix}_{suffix}.yaml; do')
    print('     cell=$(basename "$cfg" .yaml)')
    print(f'     dst="{model_dir}/$cell/{args.fold}/pretrained"; mkdir -p "$dst"')
    print(f'     for f in {" ".join(ENCODER_FILES)}; do ln -sf "$src/$f" "$dst/$f"; done')
    print('   done')
    print('\n2. Finetune every cell:')
    print(f'   for cfg in {args.output_dir}/{args.prefix}_{suffix}.yaml; do')
    print(f'     TASKS=mortality {launch} "$cfg"')
    print('   done')


if __name__ == '__main__':
    main()
