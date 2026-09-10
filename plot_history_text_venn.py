#!/usr/bin/env python3
"""Count patients by the kind of pre-admission record they carry, and draw the diagram.

Four nested sets, over patients rather than episodes -- a patient qualifies if any of their
episodes carries the record:

    all_text  a pre-admission record of every text feature
    any_text  a pre-admission record of any text feature
    history   at least one pre-admission record the model reads, i.e. the value stream
    all       every patient in the extraction, with or without a pre-admission record

These are the cohorts the experiments and the downstream analyses select on, inside the
population they are drawn from. `all_text` is drawn because the carry-forward contrast runs on
it: it needs the diagnosis list to decide whether a label was already named and the summary as
the input under test, so an episode missing either cannot contribute. The nesting is
structural -- a text record before the cutoff is itself a pre-admission record, and having
every text feature implies having one -- and the script checks it rather than assuming it.

Drawing `all` puts the denominator the other counts are read against into the figure: the
patients carrying no pre-admission record occupy the outermost ring, so the share of the
cohort each inner circle covers can be read off rather than computed from a caption.

The circles are concentric and their areas are proportional to set size, which for nested
sets is always drawable -- a radius is the square root of that set's share of the cohort, and
nested counts give nested radii. Nothing has to be solved or scaled to fit. Which text feature
an episode in the one-feature ring is missing is reported in the table rather than drawn: the
two are not nested in each other, so separating them is what would need a lens.

Usage:
    python plot_history_text_venn.py --data_dir data/ --output tables/history_text_venn.png

Folds partition the same patients, so one fold's train, val and test partitions cover the
cohort once. Sets are keyed on patient id, so passing more folds is harmless but redundant.
"""

import argparse
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from TransEHR2.data.cohorts import (has_all_historical_text, has_any_historical_text,
                                    has_any_history, has_historical_text,
                                    has_value_history)
from TransEHR2.data.preprocessing import load_dataset


# Positions in the dataset config's TEXT_FEATS list.
SUMMARY_INDEX = 0
DIAGNOSIS_INDEX = 1

TEXT_COLOUR = '#4878a8'
HISTORY_COLOUR = '#8a8a8a'
ALL_COLOUR = '#b8b8b8'

# The two text circles share a hue and differ in depth, because one contains the other: a
# second colour would read as a different category rather than as a subset.
ANY_TEXT_ALPHA = 0.30
ALL_TEXT_ALPHA = 0.62

# A ring thinner than this, in units of the cohort circle's radius, cannot hold a count, so
# the count is placed outside the figure on a leader line instead.
MIN_RING_LABEL_GAP = 0.14
# How far beyond the cohort circle such a label sits.
RING_LABEL_OFFSET = 0.20


def patient_ids(data_dir: str, fold: str, split: str, n_episodes: int) -> np.ndarray:
    """Patient id per row of the extracted arrays.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold name.
        split: Partition name.
        n_episodes: Row count of the arrays the ids must line up with.

    Returns:
        (n_episodes,) array of patient ids.

    Raises:
        ValueError: If the id file and the arrays disagree on length, which means the two are
            out of step and every set built from them would be wrong.
    """
    ids_path = os.path.join(data_dir, fold, f'{split}_ids.pkl')
    with open(ids_path, 'rb') as handle:
        episode_ids = pickle.load(handle)
    if len(episode_ids) != n_episodes:
        raise ValueError(
            f'{fold}/{split}: {n_episodes} episodes in the arrays but {len(episode_ids)} ids '
            f'in {ids_path}. The ids and the extracted arrays are out of step.'
        )
    # Ids are patient_id * 1000 + episode_number; see MixedDataReader.
    return np.asarray(episode_ids, dtype=np.int64) // 1000


def collect_partition(data_dir: str, fold: str, split: str,
                      extracted_history_len_steps=None) -> dict:
    """Patient id sets for one partition.

    Args:
        data_dir: Directory holding the fold subdirectories.
        fold: Fold name.
        split: Partition name.
        extracted_history_len_steps: Width of the history region, for datasets written before
            the layout was recorded in metadata.

    Returns:
        Dict of sets keyed 'text', 'all_text', 'summary', 'diagnosis', 'any' and
        'readable', plus 'all' for every patient seen.
    """
    dataset = load_dataset(os.path.join(data_dir, fold, split),
                           extracted_history_len_steps=extracted_history_len_steps)
    hist = dataset.max_history_len_steps
    has_summary = has_historical_text(dataset.val_masks, dataset.val_text_indicators, hist,
                                      SUMMARY_INDEX)
    has_diagnosis = has_historical_text(dataset.val_masks, dataset.val_text_indicators, hist,
                                        DIAGNOSIS_INDEX)
    has_text = has_any_historical_text(dataset.val_masks, dataset.val_text_indicators, hist)
    has_all_text = has_all_historical_text(dataset.val_masks, dataset.val_text_indicators,
                                           hist)
    has_any = has_any_history(dataset.val_masks, dataset.event_masks, hist)
    has_readable = has_value_history(dataset.val_masks, hist)

    patients = patient_ids(data_dir, fold, split, len(has_any))
    return {
        'text': set(patients[has_text].tolist()),
        'all_text': set(patients[has_all_text].tolist()),
        'summary': set(patients[has_summary].tolist()),
        'diagnosis': set(patients[has_diagnosis].tolist()),
        'any': set(patients[has_any].tolist()),
        'readable': set(patients[has_readable].tolist()),
        'all': set(patients.tolist()),
    }


def collect_sets(data_dir: str, folds, splits,
                 extracted_history_len_steps=None) -> dict:
    """Union the patient id sets over every requested partition."""
    totals = {key: set()
              for key in ('text', 'all_text', 'summary', 'diagnosis', 'any',
                          'readable', 'all')}
    seen = 0
    for fold in folds:
        for split in splits:
            path = os.path.join(data_dir, fold, split)
            if not os.path.isdir(path):
                print(f'  {fold}/{split}: not found, skipping', file=sys.stderr)
                continue
            partition = collect_partition(data_dir, fold, split,
                                          extracted_history_len_steps)
            for key, value in partition.items():
                totals[key] |= value
            seen += 1
            print(f'  {fold}/{split}: {len(partition["all"])} patients, '
                  f'{len(partition["any"])} with history')
    if seen == 0:
        raise SystemExit('No partitions were read. Check --data_dir, --folds and --splits.')
    return totals


def layout(counts: dict) -> dict:
    """Concentric circle radii for the figure, in units of the cohort circle's radius.

    The cohort circle is fixed at radius 1 and the others are sized from it by area, so a
    radius is the square root of that set's share of the cohort. All three are concentric,
    which for nested sets is exact and needs no solving: nested counts give nested radii, so
    the drawing is always to scale and every ring has an even width to label.

    Args:
        counts: Region counts from `region_counts`.

    Returns:
        Dict with 'all_r', 'history_r', 'text_r' and 'all_text_r'.
    """
    n_all = counts['all']
    scale = 1.0 / math.sqrt(n_all) if n_all else 1.0
    return {
        'all_r': 1.0,
        'history_r': math.sqrt(counts['history']) * scale,
        'text_r': math.sqrt(counts['text']) * scale,
        'all_text_r': math.sqrt(counts['all_text']) * scale,
    }


def ring_label(ax, value: int, inner_r: float, outer_r: float, side: int) -> bool:
    """Label a ring-shaped region on the vertical axis, above or below the centre.

    Args:
        ax: Axes to draw on.
        value: Count to print. A region with no patients gets no label.
        inner_r: Radius of the ring's inner boundary.
        outer_r: Radius of its outer boundary.
        side: +1 to label above the centre, -1 to label below.

    Returns:
        True if the ring was too thin to hold the count and it was placed outside the figure
        on a leader line, which the axis limits have to make room for.
    """
    if value <= 0:
        return False
    midpoint = side * 0.5 * (inner_r + outer_r)
    if outer_r - inner_r >= MIN_RING_LABEL_GAP:
        ax.text(0, midpoint, f'{value:,}', ha='center', va='center', fontsize=12,
                color='#333333')
        return False
    ax.annotate(f'{value:,}', xy=(0, midpoint),
                xytext=(0, side * (1.0 + RING_LABEL_OFFSET)), ha='center',
                va='bottom' if side > 0 else 'top', fontsize=12, color='#333333',
                arrowprops=dict(arrowstyle='-', color='#777777', linewidth=0.8))
    return True


def draw(counts: dict, output: str, title: str) -> None:
    """Render the Euler diagram and write it to `output`.

    Args:
        counts: Region counts from `region_counts`.
        output: Destination path; the extension picks the format.
        title: Figure title.
    """
    n_all, n_history = counts['all'], counts['history']
    n_text, n_all_text = counts['text'], counts['all_text']

    geometry = layout(counts)
    all_r, history_r = geometry['all_r'], geometry['history_r']
    text_r, all_text_r = geometry['text_r'], geometry['all_text_r']

    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    cohort = plt.Circle((0, 0), all_r, facecolor=ALL_COLOUR, alpha=0.16,
                        edgecolor=HISTORY_COLOUR, linewidth=1.4)
    history = plt.Circle((0, 0), history_r, facecolor=HISTORY_COLOUR, alpha=0.20,
                         edgecolor=HISTORY_COLOUR, linewidth=1.4)
    any_text = plt.Circle((0, 0), text_r, facecolor=TEXT_COLOUR, alpha=ANY_TEXT_ALPHA,
                          edgecolor=TEXT_COLOUR, linewidth=1.4)
    all_text = plt.Circle((0, 0), all_text_r, facecolor=TEXT_COLOUR, alpha=ALL_TEXT_ALPHA,
                          edgecolor=TEXT_COLOUR, linewidth=1.4)
    for patch in (cohort, history, any_text, all_text):
        ax.add_patch(patch)

    # The innermost region is a disc, so its count sits at the centre. The rings alternate
    # sides going outward, which keeps a leader line from one ring clear of the next one's
    # label: any ring can be too thin to hold its count, and which one is depends on the
    # cohort's proportions rather than on anything fixed.
    if n_all_text > 0:
        ax.text(0, 0, f'{n_all_text:,}', ha='center', va='center', fontsize=12)
    leaders = [
        ring_label(ax, counts['one_text_only'], all_text_r, text_r, +1),
        ring_label(ax, counts['history_only'], text_r, history_r, -1),
        ring_label(ax, counts['no_history'], history_r, all_r, +1),
    ]
    leader_above = leaders[0] or leaders[2]
    leader_below = leaders[1]

    handles = [
        Patch(facecolor=ALL_COLOUR, alpha=0.16, edgecolor=HISTORY_COLOUR,
              label=f'All patients  ({n_all:,})'),
        Patch(facecolor=HISTORY_COLOUR, alpha=0.20, edgecolor=HISTORY_COLOUR,
              label=f'Any pre-admission record  ({n_history:,})'),
        Patch(facecolor=TEXT_COLOUR, alpha=ANY_TEXT_ALPHA, edgecolor=TEXT_COLOUR,
              label=f'Any pre-admission text record  ({n_text:,})'),
        Patch(facecolor=TEXT_COLOUR, alpha=ALL_TEXT_ALPHA, edgecolor=TEXT_COLOUR,
              label=f'Both text features  ({n_all_text:,})'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
              frameon=False, fontsize=10, handlelength=1.4, borderpad=0.2)

    fig.text(0.5, 0.02, 'Areas are proportional to patient counts.', ha='center',
             va='bottom', fontsize=9, color='#555555')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.30 if leader_below else -1.15, 1.35 if leader_above else 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12)
    fig.subplots_adjust(bottom=0.22, top=0.96)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f'Wrote {output}')


def region_counts(sets: dict) -> dict:
    """Set sizes and the disjoint region counts the figure labels.

    `history` is the value stream, which is what the cohort predicate selects on and what the
    model reads; the either-stream count is carried as `any` and reported rather than drawn.

    Raises:
        ValueError: If any of the nested sets escapes the one containing it. Text history is
            built by intersecting the value stream's observed history with the feature
            indicators, so every containment here is structural and a violation means the
            streams or the id ordering are out of step. The all-features set is checked
            against each single-feature set as well: with the two text features this
            extraction carries it is exactly their intersection, so a discrepancy would mean
            the reduction and the indexed predicates disagree.
    """
    text, all_text = sets['text'], sets['all_text']
    history, everyone = sets['readable'], sets['all']
    for name, subset, superset, why in (
            ('text record', text, history, 'no pre-admission record the model reads'),
            ('record of every text feature', all_text, text, 'no pre-admission text record'),
            ('record of every text feature', all_text, sets['summary'],
             'no pre-admission discharge summary'),
            ('record of every text feature', all_text, sets['diagnosis'],
             'no pre-admission diagnosis description'),
    ):
        stray = subset - superset
        if stray:
            raise ValueError(
                f'{len(stray)} patients have a pre-admission {name} but {why}. The predicates '
                f'are nested by construction, so this cannot happen unless the arrays and ids '
                f'are misaligned.'
            )
    summary, diagnosis = sets['summary'], sets['diagnosis']
    return {
        'all': len(everyone),
        'history': len(history),
        'text': len(text),
        'all_text': len(all_text),
        'one_text_only': len(text - all_text),
        'history_only': len(history - text),
        'no_history': len(everyone - history),
        'any': len(sets['any']),
        'summary': len(summary),
        'diagnosis': len(diagnosis),
        'both': len(summary & diagnosis),
        'summary_only': len(summary - diagnosis),
        'diagnosis_only': len(diagnosis - summary),
    }


def report(counts: dict) -> None:
    """Print the counts as a table, with shares of the cohort."""
    total = counts['all'] or 1
    rows = [
        ('Patients', counts['all']),
        ('Any pre-admission record (the cohort)', counts['history']),
        ('  either stream, including events', counts['any']),
        ('No pre-admission record', counts['no_history']),
        ('Any pre-admission text record (the cohort)', counts['text']),
        ('Both text features (the carry-forward cohort)', counts['all_text']),
        ('One text feature only', counts['one_text_only']),
        ('  discharge summary only', counts['summary_only']),
        ('  diagnosis description only', counts['diagnosis_only']),
        ('History but no text', counts['history_only']),
        ('  any discharge summary', counts['summary']),
        ('  any diagnosis description', counts['diagnosis']),
    ]
    width = max(len(label) for label, _ in rows)
    print()
    print(f"{'':{width}}  {'n':>8}  {'%':>7}")
    print('-' * (width + 19))
    for label, value in rows:
        print(f'{label:{width}}  {value:>8,}  {100.0 * value / total:>6.1f}%')
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Count patients by pre-admission record type and draw the Euler diagram'
    )
    parser.add_argument('--data_dir', default='data',
                        help='Directory holding the fold subdirectories (default: data)')
    parser.add_argument('--folds', nargs='+', default=['fold0'],
                        help='Folds to read. One fold covers the cohort; sets are keyed on '
                             'patient id so more folds are redundant, not double counted.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Partitions within each fold (default: train val test)')
    parser.add_argument('--output', default='tables/history_text_venn.png',
                        help='Figure path; the extension picks the format')
    parser.add_argument('--csv', default=None, help='Also write the counts to this CSV')
    parser.add_argument('--title', default='', help='Figure title (default: none)')
    parser.add_argument('--extracted-history-len-steps', type=int, default=None,
                        help='Width of the history region in the extracted arrays. Only needed '
                             'for datasets written before the layout was recorded in metadata.')
    parser.add_argument('--no-figure', action='store_true',
                        help='Print the counts without drawing anything')
    args = parser.parse_args(argv)

    print(f'Reading {args.data_dir}: folds {" ".join(args.folds)}, '
          f'splits {" ".join(args.splits)}')
    sets = collect_sets(args.data_dir, args.folds, args.splits,
                        args.extracted_history_len_steps)
    counts = region_counts(sets)
    report(counts)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w') as handle:
            handle.write('region,patients\n')
            for key in ('all', 'history', 'text', 'all_text', 'one_text_only',
                        'history_only', 'no_history', 'any', 'summary', 'diagnosis',
                        'both', 'summary_only', 'diagnosis_only'):
                handle.write(f'{key},{counts[key]}\n')
        print(f'Wrote {args.csv}')

    if not args.no_figure:
        draw(counts, args.output, args.title)
    return 0


if __name__ == '__main__':
    sys.exit(main())
