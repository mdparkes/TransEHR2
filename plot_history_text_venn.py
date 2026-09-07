#!/usr/bin/env python3
"""Count patients by the kind of pre-admission record they carry, and draw the diagram.

Three sets, over patients rather than episodes -- a patient qualifies if any of their
episodes carries the record:

    summary    at least one pre-admission discharge summary
    diagnosis  at least one pre-admission diagnosis description
    any        at least one pre-admission record of any kind, value or event stream

Text records share the value stream's timestep axis and are found by intersecting that
stream's observed history with the feature's presence indicator, so `summary` and `diagnosis`
are subsets of `any` by construction. The figure is therefore an Euler diagram -- two circles
inside a third -- rather than a three-circle Venn with four empty regions. The script checks
the containment rather than assuming it.

Circle areas are proportional to set size, and the distance between the two text circles is
solved so their overlap is proportional too. When the inner pair cannot fit inside the outer
circle at that scale, they are shrunk together and the figure says so; every region carries
its own count either way.

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

from TransEHR2.data.cohorts import (has_any_history, has_historical_text,
                                    has_value_history)
from TransEHR2.data.preprocessing import load_dataset


# Positions in the dataset config's TEXT_FEATS list.
SUMMARY_INDEX = 0
DIAGNOSIS_INDEX = 1

CIRCLE_COLOURS = ('#4878a8', '#c0653a')
OUTER_COLOUR = '#8a8a8a'


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
        Dict of sets keyed 'summary', 'diagnosis', 'any', plus 'all' for every patient seen.
    """
    dataset = load_dataset(os.path.join(data_dir, fold, split),
                           extracted_history_len_steps=extracted_history_len_steps)
    hist = dataset.max_history_len_steps
    has_summary = has_historical_text(dataset.val_masks, dataset.val_text_indicators, hist,
                                      SUMMARY_INDEX)
    has_diagnosis = has_historical_text(dataset.val_masks, dataset.val_text_indicators, hist,
                                        DIAGNOSIS_INDEX)
    has_any = has_any_history(dataset.val_masks, dataset.event_masks, hist)
    has_readable = has_value_history(dataset.val_masks, hist)

    patients = patient_ids(data_dir, fold, split, len(has_any))
    return {
        'summary': set(patients[has_summary].tolist()),
        'diagnosis': set(patients[has_diagnosis].tolist()),
        'any': set(patients[has_any].tolist()),
        'readable': set(patients[has_readable].tolist()),
        'all': set(patients.tolist()),
    }


def collect_sets(data_dir: str, folds, splits,
                 extracted_history_len_steps=None) -> dict:
    """Union the patient id sets over every requested partition."""
    totals = {key: set() for key in ('summary', 'diagnosis', 'any', 'readable', 'all')}
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


def lens_area(r1: float, r2: float, d: float) -> float:
    """Area of the intersection of two circles of radii `r1`, `r2` whose centres are `d` apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    term1 = r1 ** 2 * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    term2 = r2 ** 2 * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    term3 = 0.5 * math.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    return term1 + term2 - term3


def solve_distance(r1: float, r2: float, target: float) -> float:
    """Centre distance giving an intersection of `target` area. Bisection; the area is
    monotonically decreasing in the distance, so the bracket is the full feasible range."""
    if target <= 0:
        return r1 + r2
    if target >= math.pi * min(r1, r2) ** 2:
        return abs(r1 - r2)
    low, high = abs(r1 - r2), r1 + r2
    for _ in range(80):
        mid = 0.5 * (low + high)
        if lens_area(r1, r2, mid) > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def draw(counts: dict, output: str, title: str) -> None:
    """Render the Euler diagram and write it to `output`.

    Args:
        counts: Region counts from `region_counts`.
        output: Destination path; the extension picks the format.
        title: Figure title.
    """
    n_any, n_sum, n_diag = counts['any'], counts['summary'], counts['diagnosis']
    n_both = counts['both']

    # Radii from areas, with the outer circle fixed at 1, then the centre distance that makes
    # the overlap proportional too.
    scale = 1.0 / math.sqrt(n_any) if n_any else 1.0
    outer_r = 1.0
    r_sum = math.sqrt(n_sum) * scale
    r_diag = math.sqrt(n_diag) * scale
    separation = solve_distance(r_sum, r_diag, math.pi * n_both * scale ** 2)

    # Place the pair on the x axis with their union centred in the outer circle.
    x_sum, x_diag = -separation / 2.0, separation / 2.0
    shift = -0.5 * ((x_sum - r_sum) + (x_diag + r_diag))
    x_sum += shift
    x_diag += shift

    # The pair may not fit inside the outer circle even though the sets nest, since equal
    # areas do not imply a containing arrangement. Shrink together and say so rather than
    # drawing a circle that spills outside its own superset.
    reach = max(abs(x_sum) + r_sum, abs(x_diag) + r_diag)
    to_scale = reach <= outer_r
    if not to_scale and reach > 0:
        squeeze = 0.97 * outer_r / reach
        x_sum, x_diag = x_sum * squeeze, x_diag * squeeze
        r_sum, r_diag = r_sum * squeeze, r_diag * squeeze

    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    outer = plt.Circle((0, 0), outer_r, facecolor=OUTER_COLOUR, alpha=0.18,
                       edgecolor=OUTER_COLOUR, linewidth=1.4)
    circle_sum = plt.Circle((x_sum, 0), r_sum, facecolor=CIRCLE_COLOURS[0], alpha=0.45,
                            edgecolor=CIRCLE_COLOURS[0], linewidth=1.4)
    circle_diag = plt.Circle((x_diag, 0), r_diag, facecolor=CIRCLE_COLOURS[1], alpha=0.45,
                             edgecolor=CIRCLE_COLOURS[1], linewidth=1.4)
    for patch in (outer, circle_sum, circle_diag):
        ax.add_patch(patch)

    # Disjoint region counts, each at the midpoint of its own span along y = 0, so the three
    # never collide. A region with no patients gets no label.
    spans = [
        (counts['summary_only'], x_sum - r_sum, x_diag - r_diag),
        (n_both, x_diag - r_diag, x_sum + r_sum),
        (counts['diagnosis_only'], x_sum + r_sum, x_diag + r_diag),
    ]
    for value, left, right in spans:
        if value > 0 and right > left:
            ax.text(0.5 * (left + right), 0, f'{value:,}',
                    ha='center', va='center', fontsize=12)

    # History but no text lives in the annulus. Put it in the gap below the inner pair when
    # that gap is tall enough to hold the text, and outside on a leader line when it is not.
    inner_bottom = -max(r_sum, r_diag)
    gap = inner_bottom - (-outer_r)
    if counts['any_only'] > 0:
        if gap >= 0.14 * outer_r:
            ax.text(0, 0.5 * (inner_bottom - outer_r), f"{counts['any_only']:,}",
                    ha='center', va='center', fontsize=12, color='#333333')
        else:
            y_edge = -outer_r + 0.5 * gap
            ax.annotate(f"{counts['any_only']:,}", xy=(0, y_edge),
                        xytext=(0, -outer_r - 0.20), ha='center', va='top', fontsize=12,
                        color='#333333',
                        arrowprops=dict(arrowstyle='-', color='#777777', linewidth=0.8))

    handles = [
        Patch(facecolor=OUTER_COLOUR, alpha=0.18, edgecolor=OUTER_COLOUR,
              label=f'Any pre-admission record  ({n_any:,})'),
        Patch(facecolor=CIRCLE_COLOURS[0], alpha=0.45, edgecolor=CIRCLE_COLOURS[0],
              label=f'Discharge summary  ({n_sum:,})'),
        Patch(facecolor=CIRCLE_COLOURS[1], alpha=0.45, edgecolor=CIRCLE_COLOURS[1],
              label=f'Diagnosis description  ({n_diag:,})'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
              frameon=False, fontsize=10, handlelength=1.4, borderpad=0.2)

    caption = (f"{counts['all']:,} patients; {counts['no_history']:,} with no pre-admission "
               f"record. Areas are proportional to patient counts.")
    if not to_scale:
        caption = (f"{counts['all']:,} patients; {counts['no_history']:,} with no "
                   f"pre-admission record.\nInner circles scaled to fit; their areas are not "
                   f"proportional to the outer circle.")
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=9, color='#555555')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.30, 1.15)
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

    Raises:
        ValueError: If a text set is not contained in the any-record set. Text history is
            built by intersecting the value stream's observed history with the feature
            indicator, so containment is structural and a violation means the streams or the
            id ordering are out of step.
    """
    summary, diagnosis, any_record, everyone = (
        sets['summary'], sets['diagnosis'], sets['any'], sets['all'])
    for name, subset in (('summary', summary), ('diagnosis', diagnosis)):
        stray = subset - any_record
        if stray:
            raise ValueError(
                f'{len(stray)} patients have pre-admission {name} text but no pre-admission '
                f'record of any kind. Text history is an intersection with the value stream, '
                f'so this cannot happen unless the arrays and ids are misaligned.'
            )
    both = summary & diagnosis
    return {
        'all': len(everyone),
        'any': len(any_record),
        'summary': len(summary),
        'diagnosis': len(diagnosis),
        'both': len(both),
        'summary_only': len(summary - diagnosis),
        'diagnosis_only': len(diagnosis - summary),
        'any_only': len(any_record - summary - diagnosis),
        'no_history': len(everyone - any_record),
        'readable': len(sets['readable']),
    }


def report(counts: dict) -> None:
    """Print the counts as a table, with shares of the cohort."""
    total = counts['all'] or 1
    rows = [
        ('Patients', counts['all']),
        ('Any pre-admission record', counts['any']),
        ('  readable by the model', counts['readable']),
        ('No pre-admission record', counts['no_history']),
        ('Discharge summary', counts['summary']),
        ('Diagnosis description', counts['diagnosis']),
        ('Both text features', counts['both']),
        ('Discharge summary only', counts['summary_only']),
        ('Diagnosis description only', counts['diagnosis_only']),
        ('History but no text', counts['any_only']),
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
            for key in ('all', 'any', 'no_history', 'summary', 'diagnosis', 'both',
                        'summary_only', 'diagnosis_only', 'any_only'):
                handle.write(f'{key},{counts[key]}\n')
        print(f'Wrote {args.csv}')

    if not args.no_figure:
        draw(counts, args.output, args.title)
    return 0


if __name__ == '__main__':
    sys.exit(main())
