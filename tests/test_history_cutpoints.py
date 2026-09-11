"""Probes for the history sweep's cutpoint placement.

The failure this guards against is silent: any list of descending lengths produces a sweep that
runs and reports, and nothing in the output says whether the arms are spaced in a way that
makes their differences comparable. Steps evenly spaced in `HISTORY_LEN_STEPS` are not evenly
spaced in records -- the episodes reaching back to a given depth fall off sharply -- so an
evenly spaced axis puts nearly all of the mass in its last step, and every arm above that one
differs from its neighbour by almost nothing.

The mass curve is checked against a direct slice of the same array rather than against a second
formula, since the right-justified layout is the part that is easy to get backwards: cropping to
L keeps the *last* L columns of the history region, not the first.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from choose_history_cutpoints import cutpoints, mass_curve


WIDTH = 500


def heavy_tailed(n_episodes=20000, width=WIDTH, seed=0):
    """Right-justified history with a sharply falling reach, as the extraction produces."""
    rng = np.random.default_rng(seed)
    depth = np.minimum(rng.pareto(1.2, size=n_episodes) * 12 + 1, width).astype(int)
    observed = np.zeros((n_episodes, width), dtype=bool)
    for row, reach in enumerate(depth):
        observed[row, width - reach:] = True
    return observed


# ------------------------------------------------------------------------------------------
# The mass curve
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('length', [0, 1, 7, 50, 231, 499, 500])
def test_the_curve_counts_the_most_recent_steps(length):
    """History is right-justified, so a crop keeps the end of the region."""
    observed = heavy_tailed()
    curve = mass_curve(observed)
    kept = observed[:, WIDTH - length:] if length else observed[:, :0]
    assert curve[length] == kept.sum()


def test_the_curve_spans_nothing_to_everything():
    curve = mass_curve(heavy_tailed())
    assert curve[0] == 0
    assert curve[WIDTH] == heavy_tailed().sum()
    assert len(curve) == WIDTH + 1


def test_the_curve_never_decreases():
    """A longer crop cannot retain fewer records, and a dip would mean the axis is reversed."""
    curve = mass_curve(heavy_tailed())
    assert np.all(np.diff(curve) >= 0)


# ------------------------------------------------------------------------------------------
# The cutpoints
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('n_steps', [3, 6, 9])
def test_the_endpoints_are_the_full_region_and_zero(n_steps):
    lengths = cutpoints(mass_curve(heavy_tailed()), n_steps)
    assert len(lengths) == n_steps
    assert lengths[0] == WIDTH and lengths[-1] == 0


@pytest.mark.parametrize('n_steps', [3, 6, 9])
def test_the_lengths_descend_without_repeating(n_steps):
    """A repeated length is two arms running the same configuration."""
    lengths = cutpoints(mass_curve(heavy_tailed()), n_steps)
    assert lengths == sorted(lengths, reverse=True)
    assert len(set(lengths)) == len(lengths)


@pytest.mark.parametrize('n_steps', [3, 6, 9])
def test_each_step_drops_about_the_same_mass(n_steps):
    """The property the cutpoints exist for. The tolerance is the integer grid: near the end of
    the axis a single step carries a large share, so no placement splits the mass exactly."""
    curve = mass_curve(heavy_tailed())
    lengths = cutpoints(curve, n_steps)
    drops = np.array([curve[a] - curve[b] for a, b in zip(lengths, lengths[1:])])
    shares = drops / curve[-1]
    assert shares.sum() == pytest.approx(1.0)
    assert np.abs(shares - 1.0 / (n_steps - 1)).max() < 0.02


def test_evenly_spaced_lengths_would_not_do_this():
    """Recorded because it is the alternative someone would reach for, and it fails badly: on
    this distribution a linear axis leaves the last step carrying most of the records."""
    curve = mass_curve(heavy_tailed())
    linear = [WIDTH, 400, 300, 200, 100, 0]
    drops = np.array([curve[a] - curve[b] for a, b in zip(linear, linear[1:])])
    shares = drops / curve[-1]
    assert shares.max() > 0.6
    solved = cutpoints(curve, len(linear))
    solved_drops = np.array([curve[a] - curve[b] for a, b in zip(solved, solved[1:])])
    assert (solved_drops / curve[-1]).max() < 0.25


def test_a_uniform_reach_gives_an_evenly_spaced_axis():
    """When every episode reaches the full region the mass is linear in the crop length, so the
    solved cutpoints are the evenly spaced ones -- the two rules agree exactly where they
    should, which is what shows the solver is not simply compressing toward zero."""
    observed = np.ones((100, WIDTH), dtype=bool)
    assert cutpoints(mass_curve(observed), 6) == [500, 400, 300, 200, 100, 0]


# ------------------------------------------------------------------------------------------
# Refusals
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('n_steps', [0, 1, 2])
def test_too_few_steps_is_refused(n_steps):
    with pytest.raises(SystemExit, match='interior cutpoint'):
        cutpoints(mass_curve(heavy_tailed()), n_steps)


def test_a_cohort_with_no_history_is_refused():
    """Dividing zero mass into equal shares would otherwise return every length as zero."""
    with pytest.raises(SystemExit, match='no pre-admission records'):
        cutpoints(mass_curve(np.zeros((10, WIDTH), dtype=bool)), 6)
