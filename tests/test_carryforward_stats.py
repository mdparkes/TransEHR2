"""Probes for the standard errors and the correction ratio of the carry-forward table.

Both quantities in a cell are computed from the same per-fold values, and an error in either
produces a table that looks entirely normal: the standard error is a small number next to a
mean and the P value is bounded, so neither can be read as wrong on its face.

Two specific ways they can disagree with each other. A standard error taken as sd / sqrt(k)
beside a P value from the corrected test describes a narrower interval than the test it sits
next to, so a reader who recomputes t from the cell gets a different answer from the one
printed in it. And the ratio of test set size to training set size is a property of how the
data were split, not of how many folds finished, so taking it from the paired count makes a
partial comparison conservative for a reason that has nothing to do with the data.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report_carryforward_comparison import compare, summarise
from reporting.stats import corrected_resampled_ttest, corrected_standard_error


FOLDS = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']


def arm(values, folds=None, phenotype='Sepsis', n_named=10, n_unnamed=20):
    """One arm's per-fold table, indexed the way the reader builds it."""
    folds = folds or FOLDS
    rows = [{'phenotype': phenotype, 'fold': fold, 'prob_named_higher': value,
             'n_named': n_named, 'n_unnamed': n_unnamed}
            for fold, value in zip(folds, values)]
    return pd.DataFrame(rows).set_index(['phenotype', 'fold'])


# ------------------------------------------------------------------------------------------
# The standard errors
# ------------------------------------------------------------------------------------------

def test_summarise_returns_the_corrected_standard_error():
    values = [0.61, 0.63, 0.59, 0.64, 0.60]
    mean, se = summarise(values, 1 / 4)
    assert mean == pytest.approx(np.mean(values))
    assert se == pytest.approx(corrected_standard_error(values, 1 / 4))


def test_the_corrected_se_is_wider_than_the_plain_one():
    """The whole reason for the change: the plain SEM understates it."""
    values = [0.61, 0.63, 0.59, 0.64, 0.60]
    _, se = summarise(values, 1 / 4)
    assert se == pytest.approx(1.5 * np.std(values, ddof=1) / math.sqrt(5))


def test_every_column_of_a_row_carries_the_same_kind_of_standard_error():
    """The two arm columns and the difference column are all means over the same
    overlapping training sets, so one of them reported uncorrected would be the odd
    number out with nothing saying so."""
    reference = arm([0.55, 0.57, 0.54, 0.58, 0.56])
    text = arm([0.61, 0.63, 0.59, 0.64, 0.60])
    row = compare(reference, text).iloc[0]

    for column, values in (('reference_sem', [0.55, 0.57, 0.54, 0.58, 0.56]),
                           ('text_sem', [0.61, 0.63, 0.59, 0.64, 0.60])):
        assert row[column] == pytest.approx(corrected_standard_error(values, 1 / 4))


def test_the_difference_column_lets_a_reader_recompute_the_test():
    """t = delta / delta_sem must give back the P printed in the same row."""
    reference = arm([0.55, 0.57, 0.54, 0.58, 0.56])
    text = arm([0.61, 0.63, 0.59, 0.64, 0.60])
    row = compare(reference, text).iloc[0]
    assert row['t_statistic'] == pytest.approx(row['delta'] / row['delta_sem'])


# ------------------------------------------------------------------------------------------
# The correction ratio
# ------------------------------------------------------------------------------------------

def test_the_ratio_comes_from_the_design_and_not_from_the_paired_folds():
    """One arm is missing a fold, so three folds pair. Each model still trained on
    four fifths of the data, so the ratio is 1/4 and not 1/2."""
    reference = arm([0.55, 0.57, 0.54], folds=['fold1', 'fold2', 'fold3'])
    text = arm([0.61, 0.63, 0.59, 0.64], folds=['fold1', 'fold2', 'fold3', 'fold4'])
    row = compare(reference, text).iloc[0]

    paired = [0.61 - 0.55, 0.63 - 0.57, 0.59 - 0.54]
    assert row['n_folds'] == 3
    assert row['delta_sem'] == pytest.approx(corrected_standard_error(paired, 1 / 3))
    # Four distinct folds were run between the two arms, so the ratio is 1/(4-1),
    # not 1/(3-1) from the three that paired.
    assert row['delta_sem'] != pytest.approx(corrected_standard_error(paired, 1 / 2))


def test_an_explicit_ratio_is_used_as_given():
    reference = arm([0.55, 0.57, 0.54, 0.58, 0.56])
    text = arm([0.61, 0.63, 0.59, 0.64, 0.60])
    row = compare(reference, text, n_train_test_ratio=1 / 9).iloc[0]
    expected = corrected_resampled_ttest(
        [0.61, 0.63, 0.59, 0.64, 0.60], [0.55, 0.57, 0.54, 0.58, 0.56],
        n_train_test_ratio=1 / 9)
    assert row['t_statistic'] == pytest.approx(expected.statistic)


def test_the_arms_must_share_a_fold():
    reference = arm([0.55, 0.57], folds=['fold1', 'fold2'])
    text = arm([0.61, 0.63], folds=['fold3', 'fold4'])
    with pytest.raises(ValueError):
        compare(reference, text)
