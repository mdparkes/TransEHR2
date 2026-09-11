#!/usr/bin/env python3
"""Choose the history-length sweep's cutpoints so each step drops equal record mass.

`HISTORY_LEN_STEPS` crops the extracted arrays to their most recent L pre-admission timesteps,
so the sweep's arms are nested: each one holds everything the next one down holds. What a step
costs is therefore the records it removes, and steps evenly spaced in L are not evenly spaced
in records -- the count of episodes reaching back to a given depth falls off sharply, so the
first few hundred steps of the axis carry a small share of the mass and the last few carry
most of it.

This measures the retained mass

    M(L) = the observed value-stream timesteps inside the most recent L of the history region,
           summed over the cohort

and returns the L at which M(L) crosses each equal share of M(H). The endpoints are fixed: the
longest arm is the extraction's full history region and the shortest is zero.

Only the value stream is counted. `collate_tensorized` slices the history region off the event
stream before a batch is built, so no pre-admission event reaches the model and event-stream
mass is not something a crop can remove. Text records are reported alongside as a check, since
they are a small and unevenly distributed share of the same timesteps.

Usage:
    python choose_history_cutpoints.py --data_dir data --fold fold0
    python choose_history_cutpoints.py --steps 8 --cohort any_text

One fold's train, val and test partitions cover the cohort once, so the default reads fold0 --
which is also the fold the sweep runs on.
"""

import argparse
import os
import sys

import numpy as np

from TransEHR2.data.cohorts import cohort_mask, history_observed
from TransEHR2.data.preprocessing import load_dataset


DEFAULT_SPEC = os.path.join('TransEHR2', 'configs', 'experiments', 'tuning',
                            'phase3_spec.yaml')


def collect(data_dir, fold, splits, cohort, extracted_history_len_steps=None):
    """Per-episode history occupancy over one fold, restricted to a cohort.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold name.
        splits: Partitions within the fold.
        cohort: Cohort name, or None for every episode.
        extracted_history_len_steps: Width of the history region, for datasets written before
            the layout was recorded in metadata.

    Returns:
        Dict with 'history' and 'text', each an (n_episodes, H) boolean array over the history
        region, and 'width', the region's width H.

    Raises:
        SystemExit: If no partition was read, or the partitions disagree on H.
    """
    history, text, width = [], [], None

    for split in splits:
        path = os.path.join(data_dir, fold, split)
        if not os.path.isdir(path):
            print(f'  {fold}/{split}: not found, skipping', file=sys.stderr)
            continue
        dataset = load_dataset(path, extracted_history_len_steps=extracted_history_len_steps)
        hist = int(dataset.max_history_len_steps)
        if width is None:
            width = hist
        elif hist != width:
            raise SystemExit(
                f'{fold}/{split} has a history region of {hist} steps but an earlier partition '
                f'has {width}. They are different extractions.'
            )

        keep = cohort_mask(dataset, cohort)
        observed = history_observed(dataset.val_masks, hist)
        # A text record occupies a value-stream timestep like any other, so it is a subset of
        # the same grid rather than a separate axis.
        indicators = np.asarray(dataset.val_text_indicators)[:, :hist, :]
        has_text = (indicators > 0).any(axis=2) & observed
        if keep is not None:
            observed, has_text = observed[keep], has_text[keep]
        history.append(observed)
        text.append(has_text)
        print(f'  {fold}/{split}: {observed.shape[0]} episodes, '
              f'{int(observed.sum()):,} history records')

    if not history:
        raise SystemExit('No partitions were read. Check --data_dir, --fold and --splits.')
    return {'history': np.concatenate(history, axis=0),
            'text': np.concatenate(text, axis=0),
            'width': width}


def mass_curve(observed):
    """Retained records as a function of the crop length.

    History is right-justified in `[0, H)`, so cropping to L keeps columns `[H - L, H)` and the
    curve is the reversed column sums accumulated from the most recent step backwards.

    Args:
        observed: (n_episodes, H) boolean array over the history region.

    Returns:
        (H + 1,) array where entry L is the records retained by a crop to L steps. Entry 0 is
        zero and entry H is every record in the region.
    """
    per_step = observed.sum(axis=0)[::-1]
    return np.concatenate([[0], np.cumsum(per_step)]).astype(np.int64)


def cutpoints(curve, n_steps):
    """The crop lengths that split the retained mass into equal shares.

    Args:
        curve: The array `mass_curve` returns.
        n_steps: How many values the sweep reports, endpoints included. The two endpoints are
            the full region and zero, so `n_steps - 1` shares lie between them.

    Returns:
        List of crop lengths, descending, beginning at the full region and ending at 0.

    Raises:
        SystemExit: If fewer than three values are asked for, which leaves no interior point.
    """
    if n_steps < 3:
        raise SystemExit(f'--steps {n_steps} leaves no interior cutpoint to place; use 3 or '
                         f'more.')
    width = len(curve) - 1
    total = int(curve[-1])
    if total == 0:
        raise SystemExit('The cohort holds no pre-admission records, so there is no mass to '
                         'divide.')

    lengths = [width]
    for share in range(1, n_steps - 1):
        target = total * (n_steps - 1 - share) / (n_steps - 1)
        # The first length whose retained mass is at or below the target, so a step never
        # keeps more than its share.
        length = int(np.searchsorted(curve, target, side='right') - 1)
        lengths.append(max(0, min(width, length)))
    lengths.append(0)
    return lengths


def report(lengths, curve, text_curve):
    """Print what each arm retains and what the step below it drops."""
    total = int(curve[-1])
    text_total = int(text_curve[-1])
    print(f'\n{"steps":>8}  {"records":>12}  {"share":>7}  {"dropped":>10}  {"of total":>8}  '
          f'{"text":>10}  {"share":>7}')
    print('-' * 74)
    previous = None
    for length in lengths:
        kept = int(curve[length])
        text_kept = int(text_curve[length])
        dropped = '' if previous is None else f'{previous - kept:,}'
        share = '' if previous is None else f'{(previous - kept) / total:.3f}'
        print(f'{length:>8}  {kept:>12,}  {kept / total:>7.3f}  {dropped:>10}  {share:>8}  '
              f'{text_kept:>10,}  '
              f'{(text_kept / text_total if text_total else 0):>7.3f}')
        previous = kept


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Place the history sweep cutpoints at equal drops in record mass'
    )
    parser.add_argument('--data_dir', default='data',
                        help='Directory holding the fold subdirectories (default: data)')
    parser.add_argument('--fold', default='fold0',
                        help='Fold to measure. The sweep runs on fold0 (default: fold0)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Partitions within the fold (default: train val test)')
    parser.add_argument('--cohort', default='any_history',
                        help='Cohort the sweep runs on (default: any_history), or "all"')
    parser.add_argument('--steps', type=int, default=6,
                        help='How many values the sweep reports, endpoints included '
                             '(default: 6)')
    parser.add_argument('--extracted-history-len-steps', type=int, default=None,
                        help='Width of the history region in the extracted arrays. Only needed '
                             'for datasets written before the layout was recorded in metadata.')
    args = parser.parse_args(argv)

    cohort = None if args.cohort == 'all' else args.cohort
    print(f'Reading {args.data_dir}: fold {args.fold}, splits {" ".join(args.splits)}, '
          f'cohort {args.cohort}')
    data = collect(args.data_dir, args.fold, args.splits, cohort,
                   args.extracted_history_len_steps)

    curve = mass_curve(data['history'])
    text_curve = mass_curve(data['text'])
    lengths = cutpoints(curve, args.steps)

    print(f'\n{data["history"].shape[0]:,} episodes, history region {data["width"]} steps, '
          f'{int(curve[-1]):,} records ({int(text_curve[-1]):,} carrying text)')
    report(lengths, curve, text_curve)

    print(f'\nPaste into the GRID of {DEFAULT_SPEC}:\n')
    print('  HISTORY_LEN_STEPS:')
    print(f'    values: [{", ".join(str(length) for length in lengths)}]')
    print('    select_on: mortality')
    return 0


if __name__ == '__main__':
    sys.exit(main())
