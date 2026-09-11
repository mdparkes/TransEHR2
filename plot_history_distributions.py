#!/usr/bin/env python3
"""Histogram the pre-admission record count and the gap to ICU admission.

Two panels over the episodes carrying at least one pre-admission value-associated record --
the `readable` set of `plot_history_text_venn.py`, and `has_value_history` in
`TransEHR2.data.cohorts`. Episodes are indexed to a patient's last ICU stay, one episode per
patient, so the population is the same object counted either way.

    count  value-stream pre-admission records per episode
    gap    hours from the most recent pre-admission value record to ICU admission (t = 0)

Both the population and the count are value-stream only, because that is the whole of the
history the models read: `collate_tensorized` slices the history region off the event stream
before the batch is built, so no pre-admission event reaches the encoder and the event stream
contributes only in-stay records. An episode whose sole history is an event record is therefore
outside the population rather than a zero in it; the number excluded that way is reported, so
the difference from the either-stream `any` set stays visible.

Extraction keeps at most `MAX_HISTORY_LEN_STEPS` records per stream, dropping the oldest, so
the count is right-censored. The final bin is therefore left-closed at that width and labelled
as an inequality: the mass sitting exactly at the cap is censored, not a mode. No bin above the
cap is drawn, and a count exceeding it is an error rather than a tail.

Bins are unequal in width and the bars are not, so heights are counts and the panels are not
density plots. Both distributions span four to five orders of magnitude, which no equal-width
binning resolves.

History is located by position, not by timestamp -- `[0, max_history_len_steps)`,
right-justified -- which is what `HISTORY_LEN_STEPS` crops and so agrees with what the model
can read. The timestamps are then only used for the gap, and their sign convention is checked
rather than assumed.

Usage:
    python plot_history_distributions.py --data_dir data/ \\
        --output tables/history_distributions.png --csv tables/history_distributions.csv

One fold's train, val and test partitions cover the cohort once, so the default reads fold0.
Passing more folds double counts, unlike the patient-keyed sets in plot_history_text_venn.py.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from TransEHR2.data.cohorts import (COHORTS, cohort_mask, has_any_history,
                                   has_value_history, history_observed)
from TransEHR2.data.preprocessing import load_dataset, load_episode_ids


# Figure caption per cohort, completing "n = {N} ...".
#
# The figure says "historical" where the code says pre-admission: they are the same records,
# those older than PREADMISSION_CUTOFF_HOURS before admission, and the figure's footer gives
# that bound in hours. Every name in `TransEHR2.data.cohorts.COHORTS` needs an entry --
# tests/test_history_distributions.py holds them in step, since a cohort without one raises
# only when the figure is drawn, which is after the arrays have been read.
CAPTIONS = {
    'any_history': 'episodes with at least one historical value record',
    'discharge_summary': 'episodes with at least one historical discharge summary',
    'diagnosis_history': 'episodes with at least one historical diagnosis description',
    'any_text': ('episodes with at least one historical discharge summary or discharge '
                 'diagnosis'),
    'all_text': ('episodes with both a historical discharge summary and a historical '
                 'discharge diagnosis'),
}

COUNT_COLOUR = '#4878a8'
GAP_COLOUR = '#c0653a'

# (label, lower, upper) with integer bounds inclusive. There is no zero bin: the population is
# defined by carrying at least one value record, and a bar that cannot be nonzero is noise. The
# final entry's upper is None -- it is the censored bin, and its lower bound is replaced by the
# extraction's history width so the label cannot claim a limit the data does not have.
COUNT_BINS = [
    ('1', 1, 1),
    ('2-5', 2, 5),
    ('6-10', 6, 10),
    ('11-25', 11, 25),
    ('26-50', 26, 50),
    ('51-100', 51, 100),
    ('101-250', 101, 250),
    ('251-499', 251, 499),
    (None, 500, None),
]

# (label, lower, upper) in hours, half-open [lower, upper). Months are 730 h, years 8760 h.
GAP_BINS = [
    # Every historical record is older than PREADMISSION_CUTOFF_HOURS, so this bin cannot hold
    # anything and is kept for exactly that reason: a bar here would mean a peri-stay record
    # reached the history region. Splitting it finer would spend four bars showing the same
    # emptiness.
    ('0-48 h', 0.0, 48.0),
    ('2-7 d', 48.0, 168.0),
    ('1-4 wk', 168.0, 672.0),
    ('1-6 mo', 672.0, 4380.0),
    ('6-12 mo', 4380.0, 8760.0),
    ('1-2 y', 8760.0, 17520.0),
    ('2-5 y', 17520.0, 43800.0),
    ('>5 y', 43800.0, float('inf')),
]


def patient_ids(partition_dir: str, n_episodes: int) -> np.ndarray:
    """Patient id per row of the extracted arrays.

    Args:
        partition_dir: The partition directory, e.g. `{fold}/train`.
        n_episodes: Row count of the arrays the ids must line up with.

    Returns:
        (n_episodes,) array of patient ids.
    """
    # Ids are patient_id * 1000 + episode_number; see MixedDataReader.
    return load_episode_ids(partition_dir, n_episodes) // 1000


def check_time_axis(times, masks, hist: int, label: str, cutoff_hours: float = 0.0) -> None:
    """Assert the array regions and the timestamps agree on where the eras divide.

    Every timestep in the history region must be earlier than `-cutoff_hours`, and every
    timestep in the episode region no earlier than it. With a cutoff of 0 that is the original
    check: history strictly before admission, the episode region at or after it.

    The gap is computed as a distance from zero, so it is wrong by an unbounded amount if the
    axis runs the other way or is offset. An earlier extraction had the axis inverted and every
    number derived from it was void, so this is checked on every run rather than trusted.

    `cutoff_hours` must match the `PREADMISSION_CUTOFF_HOURS` the arrays were extracted with.
    The extraction does not record it, so it is passed in; a value that is too small makes this
    raise rather than quietly mislabel an era.

    Args:
        times: (n_episodes, max_ts_len) timestamps in hours relative to ICU admission.
        masks: (n_episodes, max_ts_len) nonzero at non-padding timesteps.
        hist: Width of the history region.
        label: Stream name, for the error message.
        cutoff_hours: Hours before admission at which the episode region opens.

    Raises:
        ValueError: If either region carries timestamps belonging to the other era.
    """
    t = np.asarray(times, dtype=np.float64)
    observed = np.asarray(masks) > 0
    before = observed.copy()
    before[:, hist:] = False
    after = observed.copy()
    after[:, :hist] = False
    boundary = -float(cutoff_hours)

    if before.any() and t[before].max() >= boundary:
        raise ValueError(
            f'{label}: a pre-admission timestep carries timestamp {t[before].max():.2f}, '
            f'which is not earlier than the era boundary at {boundary:.2f} h. Either the '
            f'time axis is inverted or --extracted-cutoff-hours does not match the '
            f'PREADMISSION_CUTOFF_HOURS the arrays were extracted with.'
        )
    if after.any() and t[after].min() < boundary:
        raise ValueError(
            f'{label}: an episode-region timestep carries timestamp {t[after].min():.2f}, '
            f'which is earlier than the era boundary at {boundary:.2f} h. Either the time '
            f'axis is inverted or --extracted-cutoff-hours is larger than the '
            f'PREADMISSION_CUTOFF_HOURS the arrays were extracted with.'
        )


def latest_history_time(times, masks, hist: int) -> np.ndarray:
    """Timestamp of each episode's most recent pre-admission record in one stream.

    Args:
        times: (n_episodes, max_ts_len) timestamps in hours relative to ICU admission.
        masks: (n_episodes, max_ts_len) nonzero at non-padding timesteps.
        hist: Width of the history region.

    Returns:
        (n_episodes,) array, NaN where the stream holds no pre-admission record.
    """
    t = np.asarray(times, dtype=np.float64)[:, :hist]
    observed = history_observed(masks, hist)
    return np.where(observed.any(axis=1), np.max(np.where(observed, t, -np.inf), axis=1),
                    np.nan)


def collect_partition(data_dir: str, fold: str, split: str, cohort: str,
                      extracted_history_len_steps=None, cutoff_hours: float = 0.0) -> dict:
    """Per-episode counts, gaps and cohort flags for one partition.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold name.
        split: Partition name.
        cohort: A name in `TransEHR2.data.cohorts.COHORTS`, selecting the population.
        extracted_history_len_steps: Width of the history region, for datasets written before
            the layout was recorded in metadata.

    Returns:
        Dict of (n_episodes,) arrays plus the history width the partition was read at.
    """
    partition_dir = os.path.join(data_dir, fold, split)
    dataset = load_dataset(partition_dir,
                           extracted_history_len_steps=extracted_history_len_steps)
    hist = dataset.max_history_len_steps
    if hist <= 0:
        raise ValueError(
            f'{fold}/{split}: the arrays reserve no history region, so there is nothing to '
            f'histogram. Check --extracted-history-len-steps.'
        )

    check_time_axis(dataset.val_times, dataset.val_masks, hist, f'{fold}/{split} value',
                    cutoff_hours)
    check_time_axis(dataset.event_times, dataset.event_masks, hist,
                    f'{fold}/{split} event', cutoff_hours)

    val_observed = history_observed(dataset.val_masks, hist)
    val_count = val_observed.sum(axis=1).astype(np.int64)
    if val_count.max(initial=0) > hist:
        raise ValueError(
            f'{fold}/{split}: an episode holds {val_count.max()} pre-admission value records '
            f'but the history region is {hist} wide.'
        )

    val_latest = latest_history_time(dataset.val_times, dataset.val_masks, hist)
    event_latest = latest_history_time(dataset.event_times, dataset.event_masks, hist)

    return {
        'patient': patient_ids(partition_dir, len(val_count)),
        'val_count': val_count,
        'val_latest': val_latest,
        'event_latest': event_latest,
        'in_cohort': cohort_mask(dataset, cohort),
        'in_value': has_value_history(dataset.val_masks, hist),
        'in_any': has_any_history(dataset.val_masks, dataset.event_masks, hist),
        'hist': hist,
    }


def collect(data_dir: str, folds, splits, cohort: str,
            extracted_history_len_steps=None, cutoff_hours: float = 0.0) -> dict:
    """Concatenate the per-episode arrays over every requested partition.

    Raises:
        SystemExit: If no partition was read, or if a patient appears twice -- which means the
            partitions overlap and the histogram would weight those episodes double.
    """
    parts = []
    widths = set()
    for fold in folds:
        for split in splits:
            path = os.path.join(data_dir, fold, split)
            if not os.path.isdir(path):
                print(f'  {fold}/{split}: not found, skipping', file=sys.stderr)
                continue
            part = collect_partition(data_dir, fold, split, cohort,
                                     extracted_history_len_steps, cutoff_hours)
            widths.add(part.pop('hist'))
            parts.append(part)
            print(f'  {fold}/{split}: {len(part["val_count"])} episodes, '
                  f'{int(part["in_cohort"].sum())} in the cohort')
    if not parts:
        raise SystemExit('No partitions were read. Check --data_dir, --folds and --splits.')
    if len(widths) > 1:
        raise SystemExit(f'Partitions disagree on the history width: {sorted(widths)}.')

    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    merged['hist'] = widths.pop()

    unique, counts = np.unique(merged['patient'], return_counts=True)
    if (counts > 1).any():
        repeated = int((counts > 1).sum())
        raise SystemExit(
            f'{repeated} patients appear in more than one partition, so those episodes would '
            f'be counted twice. One fold covers the cohort once -- pass a single fold.'
        )
    return merged


def count_bins(hist: int):
    """Return the count bins with the censored bin anchored at the extraction width.

    Raises:
        SystemExit: If the history width is at or below the last closed bin's lower bound, in
            which case the labels would describe bins the data cannot fill.
    """
    closed = [entry for entry in COUNT_BINS if entry[2] is not None]
    if hist <= closed[-1][1]:
        raise SystemExit(
            f'the history region is {hist} wide, which falls inside the "{closed[-1][0]}" bin. '
            f'Edit COUNT_BINS so the censored bin opens at or below {hist}.'
        )
    return closed + [(f'\u2265{hist}', hist, None)]


def bin_counts(values: np.ndarray, bins) -> list:
    """Count values falling in each bin.

    Args:
        values: Values to bin. Integer bounds are inclusive; float bounds are half-open.
        bins: Sequence of (label, lower, upper); `upper` of None means unbounded above.

    Returns:
        List of (label, count).
    """
    out = []
    for label, lower, upper in bins:
        if upper is None:
            selected = values >= lower
        elif isinstance(upper, float):
            selected = (values >= lower) & (values < upper)
        else:
            selected = (values >= lower) & (values <= upper)
        out.append((label, int(np.count_nonzero(selected))))
    return out


def draw(count_rows, gap_rows, n_episodes: int, caption: str, output: str,
         title: str, cutoff_hours: float = 0.0) -> None:
    """Draw the two panels side by side and write the figure.

    Args:
        count_rows: (label, episodes) per record-count bin.
        gap_rows: (label, episodes) per gap bin.
        n_episodes: Cohort size, for the footer.
        caption: Cohort description, completing "n = {N} ...".
        output: Path to write.
        title: Figure title, or empty for none.
        cutoff_hours: The boundary the extraction placed between historical and peri-stay
            records. Stated in the footer, since "historical" is otherwise undefined on the
            figure and the bound is a setting rather than a convention.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    for ax, rows, colour, xlabel in (
        (axes[0], count_rows, COUNT_COLOUR, 'Historical records'),
        (axes[1], gap_rows, GAP_COLOUR, 'Most recent historical record to ICU admission'),
    ):
        labels = [label for label, _ in rows]
        heights = [value for _, value in rows]
        positions = np.arange(len(rows))
        ax.bar(positions, heights, width=0.82, color=colour, edgecolor='white', linewidth=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Episodes')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dcdcdc', linewidth=0.6)
        ax.set_axisbelow(True)
        headroom = max(heights) if heights else 1
        ax.set_ylim(0, headroom * 1.12)
        for position, value in zip(positions, heights):
            if value:
                ax.text(position, value + headroom * 0.015, f'{value:,}',
                        ha='center', va='bottom', fontsize=7.5)

    if title:
        fig.suptitle(title)
    footer = f'n = {n_episodes:,} {caption}'
    if cutoff_hours > 0:
        footer += (f'. Historical records are those collected more than '
                   f'{cutoff_hours:g} hours before ICU admission')
    fig.text(0.5, -0.02, footer,
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f'Wrote {output}')


def report(data: dict, count_rows, gap_rows, gap_values: np.ndarray,
           cohort: str, n_value: int, n_any: int) -> None:
    """Print the population, the two binnings and the quantiles behind them.

    The two panels have different denominators whenever `--gap_stream any` reaches past the value
    stream, so each set of shares is taken against its own panel's total.
    """
    n = len(data['val_count'])
    n_gap = len(gap_values)

    print()
    print(f'Cohort                                         : {cohort}')
    print(f'Episodes in the cohort                         : {n:,}')
    print(f'  of the value-history population              : {n_value:,}')
    print(f'  of the either-stream population              : {n_any:,}')
    print(f'History region width (records per stream)      : {data["hist"]}')
    if n_gap != n:
        print(f'Episodes with a measurable gap                 : {n_gap:,}')

    counts = data['val_count']
    print()
    print('Value-stream pre-admission records per episode')
    quantiles = np.percentile(counts, [50, 75, 90, 95, 99])
    print('  median {:.0f}, p75 {:.0f}, p90 {:.0f}, p95 {:.0f}, p99 {:.0f}, max {:d}'.format(
        *quantiles, int(counts.max())))
    print(f'  at the {data["hist"]}-record cap (censored)      : '
          f'{int(np.count_nonzero(counts >= data["hist"])):,}')
    width = max(len(label) for label, _ in count_rows)
    for label, value in count_rows:
        print(f'  {label:>{width}}  {value:>8,}  {100.0 * value / n:>5.1f}%')

    print()
    print('Hours from the most recent pre-admission record to ICU admission')
    quantiles = np.percentile(gap_values, [25, 50, 75, 90, 99])
    print('  p25 {:.1f}, median {:.1f}, p75 {:.1f}, p90 {:.1f}, p99 {:.1f}, max {:.1f}'.format(
        *quantiles, float(gap_values.max())))
    width = max(len(label) for label, _ in gap_rows)
    for label, value in gap_rows:
        print(f'  {label:>{width}}  {value:>8,}  {100.0 * value / n_gap:>5.1f}%')
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Histogram pre-admission record counts and the gap to ICU admission'
    )
    parser.add_argument('--data_dir', default='data',
                        help='Directory holding the fold subdirectories (default: data)')
    parser.add_argument('--folds', nargs='+', default=['fold0'],
                        help='Folds to read. One fold covers the cohort once; passing more '
                             'double counts, and the run stops if it detects that.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Partitions within each fold (default: train val test)')
    parser.add_argument('--cohort', choices=list(COHORTS), default='any_text',
                        help='Population, from TransEHR2.data.cohorts. The default is the '
                             'cohort the text-arm experiments run on, so the figure describes '
                             'the population its tables report. "any_history" is the '
                             'value-history set -- it resolves to has_value_history, not to '
                             'has_any_history, because the models read no historical event. '
                             '"discharge_summary" and "diagnosis_history" narrow it to '
                             'episodes carrying at least one historical record of that text '
                             'feature.')
    parser.add_argument('--gap_stream', choices=('value', 'any'), default='value',
                        help='Which stream supplies the most recent pre-admission record. '
                             '"value" is what the models read and matches the population. '
                             '"any" takes the later of the two streams, which can be an event '
                             'record the models never see.')
    parser.add_argument('--output', default='tables/history_distributions.png',
                        help='Figure path; the extension picks the format')
    parser.add_argument('--csv', default=None, help='Also write the bin counts to this CSV')
    parser.add_argument('--title', default='', help='Figure title (default: none)')
    parser.add_argument('--extracted-cutoff-hours', type=float, default=0.0,
                        help='Hours before admission at which the episode region opens in '
                             'the extracted arrays. Must match the extraction\'s '
                             'PREADMISSION_CUTOFF_HOURS, which the arrays do not record; a '
                             'mismatch stops the run rather than mislabelling an era.')
    parser.add_argument('--extracted-history-len-steps', type=int, default=None,
                        help='Width of the history region in the extracted arrays. Only needed '
                             'for datasets written before the layout was recorded in metadata.')
    parser.add_argument('--expect_n', type=int, default=None,
                        help='Population size to check against, i.e. the "readable by the '
                             'model" line of plot_history_text_venn.py. Mismatches stop the '
                             'run.')
    parser.add_argument('--no-figure', action='store_true',
                        help='Print the binnings without drawing anything')
    args = parser.parse_args(argv)

    print(f'Reading {args.data_dir}: folds {" ".join(args.folds)}, '
          f'splits {" ".join(args.splits)}')
    data = collect(args.data_dir, args.folds, args.splits, args.cohort,
                   args.extracted_history_len_steps, args.extracted_cutoff_hours)

    n_value = int(data['in_value'].sum())
    n_any = int(data['in_any'].sum())
    keep = data['in_cohort']
    data = {key: (value[keep] if isinstance(value, np.ndarray) else value)
            for key, value in data.items()}
    n = int(keep.sum())
    if n == 0:
        raise SystemExit(
            f'the {args.cohort} cohort is empty in {args.data_dir}. Check that the extraction '
            f'carries the text feature the cohort selects on.'
        )
    if int(data['val_count'].min()) < 1:
        raise SystemExit('an episode in the population holds no value record, so the count '
                         'bins do not start low enough.')
    if args.expect_n is not None and n != args.expect_n:
        raise SystemExit(
            f'the population holds {n:,} episodes but --expect_n is {args.expect_n:,}. '
            f'Reconcile before reporting anything from this run.'
        )

    if args.gap_stream == 'value':
        latest = data['val_latest']
    else:
        latest = np.fmax(data['val_latest'], data['event_latest'])
    gap_values = -latest[~np.isnan(latest)]
    undefined = n - len(gap_values)
    if undefined:
        print(f'  {undefined:,} episodes have no {args.gap_stream}-stream history and are '
              f'dropped from the gap panel', file=sys.stderr)
    if gap_values.size == 0:
        raise SystemExit('No episode has a measurable gap. Check --gap_stream.')
    if gap_values.min() < 0.0:
        raise SystemExit(f'a gap came out negative ({gap_values.min():.2f} h), so the most '
                         f'recent history record sits at or after admission.')

    count_rows = bin_counts(data['val_count'], count_bins(data['hist']))
    gap_rows = bin_counts(gap_values, GAP_BINS)
    report(data, count_rows, gap_rows, gap_values, args.cohort, n_value, n_any)

    if sum(value for _, value in count_rows) != n:
        raise SystemExit('the count bins do not cover every episode.')
    if sum(value for _, value in gap_rows) != len(gap_values):
        raise SystemExit('the gap bins do not cover every measurable gap.')

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w') as handle:
            handle.write('panel,bin,episodes\n')
            for label, value in count_rows:
                handle.write(f'value_history_records,{label},{value}\n')
            for label, value in gap_rows:
                handle.write(f'gap_hours,{label},{value}\n')
        print(f'Wrote {args.csv}')

    if not args.no_figure:
        draw(count_rows, gap_rows, n, CAPTIONS[args.cohort], args.output,
             args.title, args.extracted_cutoff_hours)
    return 0


if __name__ == '__main__':
    sys.exit(main())
