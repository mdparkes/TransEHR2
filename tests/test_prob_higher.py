"""Probes for the effect measure behind the carry-forward comparison.

`prob_higher` is the Mann-Whitney U statistic scaled to the unit interval: the probability that
a score drawn from one group exceeds one drawn from the other, ties counted as half. It is what
Table 5 reports and what the corrected resampled t test is run on, so an error in it is an error
in every number of that table -- and it would not look like one. The statistic is bounded in
[0, 1] and centred at 0.5 whatever it computes, so a wrong tie rule or a dropped correction
still produces a plausible column.

The implementation ranks a pooled array and subtracts the minimum possible rank sum, which is
the standard identity rather than the definition. These check it against the definition -- every
pair compared directly -- because the two agree only if the midranks are right, and ties are
common here: a saturating classifier returns the same score for many stays.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stratify_predictions_by_history import prob_higher


def pairwise(a, b):
    """P(a > b) + P(a == b) / 2, compared pair by pair.

    The definition the statistic is named for, written without ranks so that it shares no
    machinery with the implementation it checks.
    """
    left = np.asarray(a, dtype=float)[:, None]
    right = np.asarray(b, dtype=float)[None, :]
    return float(((left > right).sum() + 0.5 * (left == right).sum()) / left.size / right.size)


# ------------------------------------------------------------------------------------------
# Agreement with the definition
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('seed', range(20))
def test_it_matches_the_pairwise_definition_with_heavy_ties(seed):
    """Scores are rounded onto few distinct levels, which is what a saturating classifier
    produces and what makes the midrank correction bite."""
    rng = np.random.default_rng(seed)
    levels = int(rng.integers(1, 6))
    a = rng.integers(0, levels, int(rng.integers(1, 40))).astype(float)
    b = rng.integers(0, levels, int(rng.integers(1, 40))).astype(float)
    assert prob_higher(a, b) == pytest.approx(pairwise(a, b), abs=1e-12)


@pytest.mark.parametrize('seed', range(10))
def test_it_matches_the_pairwise_definition_without_ties(seed):
    rng = np.random.default_rng(100 + seed)
    a, b = rng.normal(size=37), rng.normal(size=53)
    assert prob_higher(a, b) == pytest.approx(pairwise(a, b), abs=1e-12)


# ------------------------------------------------------------------------------------------
# The properties the table's reading depends on
# ------------------------------------------------------------------------------------------

def test_identical_groups_give_exactly_one_half():
    """0.5 is the caption's "does not distinguish them", so it has to be exact rather than
    nearly so -- a tie rule that credited a full point instead of half would give 1.0 here."""
    assert prob_higher(np.full(10, 0.7), np.full(13, 0.7)) == 0.5


def test_separated_groups_give_exactly_one_and_zero():
    assert prob_higher([0.9, 0.8, 0.7], [0.2, 0.1]) == 1.0
    assert prob_higher([0.2, 0.1], [0.9, 0.8, 0.7]) == 0.0


@pytest.mark.parametrize('seed', range(20))
def test_swapping_the_groups_reflects_it_about_one_half(seed):
    """Exact antisymmetry. Table 5 subtracts one arm's value from another's, so a statistic
    that lost half a tie in one direction would bias every difference in the table."""
    rng = np.random.default_rng(200 + seed)
    a = rng.integers(0, 4, int(rng.integers(1, 20))).astype(float)
    b = rng.integers(0, 4, int(rng.integers(1, 20))).astype(float)
    assert prob_higher(a, b) + prob_higher(b, a) == pytest.approx(1.0, abs=1e-12)


def test_it_is_invariant_under_monotone_recalibration():
    """The caption claims the measure is unaffected by recalibration, which is what lets two
    models be compared on it at all. A rank statistic has that property; a mean difference
    would not."""
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=40), rng.normal(size=45)
    for transform in (lambda x: 1.0 / (1.0 + np.exp(-x)), np.exp, lambda x: 3.0 * x + 5.0):
        assert prob_higher(transform(a), transform(b)) == pytest.approx(
            prob_higher(a, b), abs=1e-12)


def test_a_single_score_in_each_group_is_a_straight_comparison():
    assert prob_higher([0.6], [0.4]) == 1.0
    assert prob_higher([0.4], [0.6]) == 0.0
    assert prob_higher([0.5], [0.5]) == 0.5


# ------------------------------------------------------------------------------------------
# Degenerate input
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('a,b', [([], [1.0]), ([1.0], []), ([], [])])
def test_an_empty_group_is_nan_rather_than_a_half(a, b):
    """A diagnosis no stay carries, or one every stay's text names, leaves a group empty. nan
    drops out of the fold average; 0.5 would enter it as evidence of no effect."""
    assert np.isnan(prob_higher(a, b))
