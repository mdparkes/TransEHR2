#!/usr/bin/env python3
"""Report the carry-forward contrast between a text arm and its in-stay-only
reference, in JMIR table format.

`stratify_predictions_by_history.py` measures, for one model, the probability
that a true-positive label whose diagnosis is already named in the patient's
earlier diagnosis text receives a higher score than one that is not:

    P(named > unnamed)

That quantity is not interpretable on its own. Whether a label is named is not
randomly assigned -- it selects for the chronic conditions the in-stay records
also reveal -- so a model that never reads the historical text still scores
named labels above unnamed ones. The reference arm measures exactly that floor,
and only the excess over it can be attributed to the text:

    delta = P(text arm) - P(in-stay-only arm)

Both arms run on the same cohort, the same episodes and the same folds, so the
difference is taken per fold and then averaged. Pairing on the fold removes the
between-fold variance that the two arms share, which a difference of two
independently averaged means would leave in.

Each diagnosis's delta is tested against zero with the corrected resampled t
test of Nadeau and Bengio (2003), which inflates the variance of the mean
per-fold difference to account for the overlap between the folds' training
sets; an ordinary paired t test is anti-conservative here. The resulting P
values are adjusted across the individual diagnoses with the Benjamini-Hochberg
procedure. The pooled row is an aggregate of the same data rather than an
independent hypothesis, so it is tested but kept out of that family.

Reads the `carryforward_{experiment}_{split}_per_fold.csv` files written by
`stratify_predictions_by_history.py --output-dir`.

Usage:
    python report_carryforward_comparison.py
    python report_carryforward_comparison.py --table-number S2 \
        --output tables/tableS2_carryforward.docx --with-counts
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from generate_redo_configs import experiment_name
from reporting.jmir.formatting import fmt_cell, fmt_number, fmt_p_value
from reporting.jmir.tables import Table, build_document, render_text
from reporting.stats import benjamini_hochberg, corrected_resampled_ttest

DEFAULT_INPUT_DIR = os.path.join('misc', 'stratified_carryforward')

# The peri-stay-only reference and the arm that adds historical text to it. Named by number
# rather than spelled out, so the experiment configs stay the one place a name is decided.
DEFAULT_REFERENCE = experiment_name(20)
DEFAULT_TEXT_ARM = experiment_name(24)
POOLED_ROW = 'All diagnoses pooled'

DEFAULT_CAPTION = (
    'Whether the model scores a diagnosis more highly when the patient’s '
    'earlier admissions already name it, with and without the historical '
    'diagnosis text as an input.'
)


def read_per_fold(input_dir, experiment, split):
    """Read one experiment's per-fold stratification table.

    Args:
        input_dir: Directory holding the stratification CSVs.
        experiment: EXPERIMENT_NAME of the arm.
        split: Split the stratification was run on.

    Returns:
        DataFrame indexed by (phenotype, fold).

    Raises:
        FileNotFoundError: If the arm has no per-fold file.
    """
    path = os.path.join(input_dir, f'carryforward_{experiment}_{split}_per_fold.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no per-fold stratification for {experiment} at {path}. Run '
            'stratify_predictions_by_history.py for this arm with --output-dir first.'
        )
    df = pd.read_csv(path)
    missing = {'phenotype', 'fold', 'prob_named_higher'} - set(df.columns)
    if missing:
        raise ValueError(f'{path} is missing columns {sorted(missing)}')
    return df.set_index(['phenotype', 'fold'])


def summarise(values):
    """Mean and standard error of the mean, ignoring nans."""
    values = np.asarray([v for v in values if v == v], dtype=float)
    if values.size == 0:
        return float('nan'), float('nan')
    if values.size == 1:
        return float(values[0]), float('nan')
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def compare(reference, text_arm, n_train_test_ratio=None):
    """Pair the two arms fold by fold and summarise the contrast.

    Args:
        reference: Per-fold table of the in-stay-only arm.
        text_arm: Per-fold table of the arm that reads the historical text.
        n_train_test_ratio: n_test / n_train for the corrected resampled t
            test. None uses the fixed 1 / (k - 1) adjustment appropriate to a
            single run of k-fold cross-validation.

    Returns:
        DataFrame with one row per diagnosis.

    Raises:
        ValueError: If the arms disagree on how many positives are named,
            which means they did not run on the same episodes.
    """
    shared_folds = sorted(
        set(reference.index.get_level_values('fold'))
        & set(text_arm.index.get_level_values('fold'))
    )
    if not shared_folds:
        raise ValueError('the two arms share no folds, so they cannot be paired.')

    rows = []
    phenotypes = [p for p in text_arm.index.get_level_values('phenotype').unique()
                  if p in set(reference.index.get_level_values('phenotype'))]

    for phenotype in phenotypes:
        ref_values, arm_values = [], []
        n_named = n_unnamed = 0
        for fold in shared_folds:
            try:
                ref_row = reference.loc[(phenotype, fold)]
                arm_row = text_arm.loc[(phenotype, fold)]
            except KeyError:
                continue

            # Named-ness is a property of the stay and the audit, not of the
            # model, so the two arms must agree on the group sizes. If they do
            # not, they ran on different episodes and the pairing is invalid.
            for column in ('n_named', 'n_unnamed'):
                if column in reference.columns and column in text_arm.columns:
                    if int(ref_row[column]) != int(arm_row[column]):
                        raise ValueError(
                            f'{phenotype} in {fold}: the reference arm has '
                            f'{int(ref_row[column])} {column} but the text arm has '
                            f'{int(arm_row[column])}. The arms did not run on the same '
                            'episodes, so their difference is not a paired contrast. '
                            'Check that both used the same cohort and the same audit.'
                        )
            n_named = int(arm_row.get('n_named', 0))
            n_unnamed = int(arm_row.get('n_unnamed', 0))

            ref_p = float(ref_row['prob_named_higher'])
            arm_p = float(arm_row['prob_named_higher'])
            # Only folds where both arms are defined can be paired, and every
            # reported quantity is computed over the same paired folds so that
            # the two arm columns and the difference describe one set.
            if ref_p == ref_p and arm_p == arm_p:
                ref_values.append(ref_p)
                arm_values.append(arm_p)

        deltas = [a - r for a, r in zip(arm_values, ref_values)]
        ref_mean, ref_sem = summarise(ref_values)
        arm_mean, arm_sem = summarise(arm_values)
        delta_mean, delta_sem = summarise(deltas)

        if len(deltas) >= 2:
            test = corrected_resampled_ttest(arm_values, ref_values,
                                             n_train_test_ratio)
            statistic, df, p_value, note = (test.statistic, test.df,
                                            test.p_value, test.note)
            # The standard error the test actually used. Reporting the plain
            # SEM beside a corrected P value would let a reader recompute t and
            # get a different answer from the one in the same cell.
            k = len(deltas)
            ratio = (1.0 / (k - 1) if n_train_test_ratio is None
                     else n_train_test_ratio)
            corrected_sem = float(np.sqrt((1.0 / k + ratio)
                                          * np.var(deltas, ddof=1)))
        else:
            statistic = df = p_value = corrected_sem = float('nan')
            note = 'fewer than two folds where both arms are defined'

        rows.append({
            'phenotype': phenotype,
            'n_named': n_named,
            'n_unnamed': n_unnamed,
            'reference_p': ref_mean, 'reference_sem': ref_sem,
            'text_p': arm_mean, 'text_sem': arm_sem,
            'delta': delta_mean, 'delta_sem': delta_sem,
            'delta_sem_corrected': corrected_sem,
            't_statistic': statistic, 'df': df,
            'p_value': p_value, 'note': note,
            'n_folds': len(deltas),
        })

    result = pd.DataFrame(rows)

    # The false discovery rate is controlled over the individual diagnoses.
    # The pooled row restates the same observations at a coarser grain, so
    # including it would count one body of evidence twice and inflate the
    # family; it keeps its unadjusted P value.
    individual = result['phenotype'] != POOLED_ROW
    result['p_adjusted'] = float('nan')
    result.loc[individual, 'p_adjusted'] = benjamini_hochberg(
        result.loc[individual, 'p_value'].tolist()
    )

    pooled = result[~individual]
    others = result[individual].sort_values('delta', ascending=False)
    return pd.concat([pooled, others], ignore_index=True)


def estimate(mean, sem, p=None, alpha=0.05, raw_p=None):
    """Format a mean with its SE and optional P value, in JMIR house style.

    Args:
        mean: Mean over folds.
        sem: Standard error of that mean.
        p: P value to report, or None for a column carrying no comparison.
        alpha: Significance level, used by the P value formatter.
        raw_p: Unadjusted P value to show alongside the adjusted one, or None.

    Returns:
        A string such as "0.845 (SE 0.004; P=.03)", or an em dash.
    """
    if mean != mean:
        return chr(8212)
    if p is None or p != p:
        return fmt_cell(mean, sem, None)
    cell = fmt_cell(mean, sem, p, alpha=alpha)
    if raw_p is not None and raw_p == raw_p:
        cell = cell[:-1] + '; unadjusted ' + fmt_p_value(raw_p, alpha) + ')'
    return cell


def build_table(result, table_number, caption, reference_name, text_arm_name,
                with_counts, alpha=0.05, show_raw_p=False):
    """Assemble the comparison table."""
    table = Table(
        number=table_number,
        caption=caption,
        stub_head='Diagnosis',
        columns=(
            (['Positives named/unnamed, n'] if with_counts else [])
            + ['In-stay records only, P(named>unnamed)',
               'In-stay records and historical text, P(named>unnamed)',
               'Difference, text arm minus in-stay only']
        ),
        short_columns=(
            (['named/unnam'] if with_counts else [])
            + ['in-stay only', 'plus text', 'difference']
        ),
    )

    statistic = table.add_footnote(
        'Among the true-positive labels of stays whose earlier admissions left '
        'diagnosis text, P(named>unnamed) is the probability that a label the '
        'text already names receives a higher score than one it does not, ties '
        'counted as one half. It is the Mann-Whitney statistic scaled to the '
        'unit interval, so it is free of any decision threshold and unchanged '
        'by monotone recalibration. Values are the mean over cross-validation '
        'folds with the standard error of that mean in parentheses.'
    )
    if with_counts:
        counts = table.add_footnote(
            'True-positive labels whose diagnosis the earlier text does and does '
            'not name, summed over folds. Both arms run on the same episodes, so '
            'the two arms share these counts.'
        )
    reference_note = table.add_footnote(
        f'{reference_name}. This arm never reads the historical diagnosis text, '
        'so it measures how far P(named>unnamed) departs from 0.5 for reasons '
        'that have nothing to do with the text: whether a label is named is not '
        'randomly assigned, and selects for the chronic conditions the in-stay '
        'records reveal on their own. It is the floor, not a null result.'
    )
    text_note = table.add_footnote(
        f'{text_arm_name}. Identical to the reference arm except that the '
        'historical diagnosis text is among its inputs.'
    )
    n_tested = int((result['phenotype'] != POOLED_ROW).sum())
    difference_note = table.add_footnote(
        'The text arm minus the reference arm, taken within each fold and then '
        'averaged, since both arms run on the same episodes in the same folds. '
        'This is the part of the association attributable to the text. A '
        'difference near zero means the model does not use text that demonstrably '
        'names the label. P values are from the corrected resampled t test of '
        'Nadeau and Bengio (2003), which inflates the variance of the mean '
        'per-fold difference by the ratio of test set size to training set size; '
        'an ordinary paired t test is anticonservative here because the folds '
        'share training data. The standard error shown is the corrected one the '
        'test uses, so it is larger than the plain standard error of the '
        'per-fold differences. P values are adjusted across the '
        + str(n_tested) + ' individual diagnoses with the Benjamini-Hochberg '
        'procedure. The pooled row restates the same observations at a coarser '
        'grain rather than testing a further hypothesis, so it is kept out of '
        'that family and its P value is unadjusted.'
    )

    table.stub_head += f'<sup>{statistic}</sup>'
    offset = 0
    if with_counts:
        table.columns[0] += f'<sup>{counts}</sup>'
        offset = 1
    table.columns[offset] += f'<sup>{reference_note}</sup>'
    table.columns[offset + 1] += f'<sup>{text_note}</sup>'
    table.columns[offset + 2] += f'<sup>{difference_note}</sup>'

    def cells(row, pooled_row=False):
        values = []
        if with_counts:
            values.append(f'{int(row.n_named):,}/{int(row.n_unnamed):,}')
        # The pooled row sits outside the Benjamini-Hochberg family, so it
        # reports its unadjusted P value directly.
        reported = row.p_value if pooled_row else row.p_adjusted
        raw = None if pooled_row or not show_raw_p else row.p_value
        values += [
            estimate(row.reference_p, row.reference_sem),
            estimate(row.text_p, row.text_sem),
            estimate(row.delta, row.delta_sem_corrected, reported, alpha, raw),
        ]
        return values

    pooled = result[result['phenotype'] == POOLED_ROW]
    if len(pooled):
        table.add_category('All diagnoses pooled')
        for row in pooled.itertuples(index=False):
            table.add_row('Any diagnosis', cells(row, pooled_row=True), level=1)

    table.add_category('Individual diagnoses')
    for row in result[result['phenotype'] != POOLED_ROW].itertuples(index=False):
        table.add_row(row.phenotype, cells(row), level=1)

    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                        help=f'Stratification output directory '
                             f'(default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--reference', default=DEFAULT_REFERENCE,
                        help='EXPERIMENT_NAME of the in-stay-only arm')
    parser.add_argument('--text-arm', default=DEFAULT_TEXT_ARM,
                        help='EXPERIMENT_NAME of the arm reading historical text')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'],
                        help='Split the stratification was run on')
    parser.add_argument('--table-number', default='S2',
                        help='Table number used in the caption (default: S2)')
    parser.add_argument('--caption', default=DEFAULT_CAPTION,
                        help='Caption text, without the "Table N." prefix')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level, used when formatting P values')
    parser.add_argument('--show-raw-p', action='store_true',
                        help='Report the unadjusted P value alongside the adjusted one')
    parser.add_argument('--n-train-test-ratio', type=float, default=None,
                        help='n_test / n_train for the corrected resampled t test. '
                             'Default uses the fixed 1/(k-1) adjustment for a single '
                             'run of k-fold cross-validation.')
    parser.add_argument('--csv', default=None,
                        help='Optional path for the underlying numbers, including '
                             'the t statistics and unadjusted P values')
    parser.add_argument('--drop-untestable', action='store_true',
                        help='Omit diagnoses with too few named or unnamed positives '
                             'to yield a difference. They are always excluded from the '
                             'Benjamini-Hochberg family regardless of this flag.')
    parser.add_argument('--with-counts', action='store_true',
                        help='Add a column of named/unnamed positive counts')
    parser.add_argument('--output', default=None,
                        help='Optional .docx path to write the table to')
    parser.add_argument('--caption-style', default=None,
                        help='Named Word style for captions')
    args = parser.parse_args(argv)

    reference = read_per_fold(args.input_dir, args.reference, args.split)
    text_arm = read_per_fold(args.input_dir, args.text_arm, args.split)
    result = compare(reference, text_arm, args.n_train_test_ratio)

    if args.drop_untestable:
        testable = result['delta'].notna() | (result['phenotype'] == POOLED_ROW)
        dropped = int((~testable).sum())
        result = result[testable].reset_index(drop=True)
        if dropped:
            print(f'Omitted {dropped} diagnoses with no testable difference',
                  file=sys.stderr)

    table = build_table(result, args.table_number, args.caption,
                        args.reference, args.text_arm, args.with_counts,
                        args.alpha, args.show_raw_p)
    render_text(table)

    notes = result[result['note'].notna() & (result['note'] != '')]
    for row in notes.itertuples(index=False):
        print(f'  note: {row.phenotype}: {row.note}', file=sys.stderr)

    if args.csv:
        directory = os.path.dirname(args.csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        result.to_csv(args.csv, index=False)
        print(f'Wrote {args.csv}')

    if args.output:
        directory = os.path.dirname(args.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        build_document(table, args.output, caption_style=args.caption_style)
        print(f'\nWrote {args.output}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
