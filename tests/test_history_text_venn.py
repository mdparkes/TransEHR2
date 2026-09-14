"""Probes for the patient-count Euler diagram.

Four things here can be silently wrong. The area solver returns a number for any input, so a
dropped factor produces a plausible-looking figure with the wrong overlap -- an earlier
revision lost a factor of pi exactly that way. The containments are structural, not incidental:
text history is built by intersecting the value stream's observed history with the feature
indicators, so a violation means the arrays and the id lists are out of step and every count is
untrustworthy. The layout has to keep every circle inside its superset, which area
proportionality alone does not guarantee -- a pair of equal-area circles can still be arranged
so one crosses the boundary of the circle containing its set. And the drawn regions have to
agree with the reported breakdown, since the lens is the both-features set the carry-forward
analysis runs on.

The two text features are not nested in each other, which is why the pair is drawn overlapping
rather than concentric: their union is the any-text cohort and their intersection is the
both-features one, so two circles carry four sets.
"""

import math

import pytest

from plot_history_text_venn import (MIN_RING_LABEL_GAP, layout, lens_area, region_counts,
                                    solve_distance)


def _sets(all_n, history_n, text_n, summary_n=None, diagnosis_n=None, any_n=None):
    """Nested patient id sets of the given sizes.

    The two feature sets cover the text set exactly and overlap in its middle, so `all_text` is
    genuinely their intersection -- which is what `region_counts` checks, and what an
    independently chosen `all_text` would violate.
    """
    summary_n = text_n if summary_n is None else summary_n
    diagnosis_n = text_n if diagnosis_n is None else diagnosis_n
    summary = set(range(summary_n))
    diagnosis = set(range(text_n - diagnosis_n, text_n))
    return {
        'all': set(range(all_n)),
        'readable': set(range(history_n)),
        'any': set(range(history_n if any_n is None else any_n)),
        'text': summary | diagnosis,
        'all_text': summary & diagnosis,
        'summary': summary,
        'diagnosis': diagnosis,
    }


# ------------------------------------------------------------------------------------------
# The area solver
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('r1,r2,fraction', [
    (1.0, 1.0, 0.5), (1.0, 0.6, 0.3), (0.8, 0.8, 0.9), (1.0, 0.3, 0.05), (0.5, 1.2, 0.75),
])
def test_the_solver_recovers_the_overlap_it_was_asked_for(r1, r2, fraction):
    """A dropped constant here changes the figure without changing anything visibly wrong."""
    target = fraction * math.pi * min(r1, r2) ** 2
    distance = solve_distance(r1, r2, target)
    assert lens_area(r1, r2, distance) == pytest.approx(target, abs=1e-9)


def test_disjoint_circles_are_placed_apart_and_contained_ones_together():
    assert solve_distance(1.0, 0.5, 0.0) == pytest.approx(1.5)
    assert solve_distance(1.0, 0.5, math.pi * 0.25) == pytest.approx(0.5)


# ------------------------------------------------------------------------------------------
# The regions
# ------------------------------------------------------------------------------------------

def test_the_regions_partition_the_cohort():
    counts = region_counts(_sets(1000, 700, 400, summary_n=300, diagnosis_n=250))
    assert counts['all_text'] + counts['one_text_only'] == counts['text']
    assert counts['text'] + counts['history_only'] == counts['history']
    assert counts['history'] + counts['no_history'] == counts['all']


def test_the_three_drawn_text_regions_partition_the_text_set():
    """The lens and the two crescents are what the figure labels, so they have to add up."""
    counts = region_counts(_sets(1000, 700, 400, summary_n=300, diagnosis_n=250))
    assert (counts['summary_only'] + counts['all_text']
            + counts['diagnosis_only']) == counts['text']
    assert counts['summary_only'] + counts['all_text'] == counts['summary']
    assert counts['diagnosis_only'] + counts['all_text'] == counts['diagnosis']


def test_the_either_stream_count_is_carried_separately():
    """`any` is descriptive and may exceed the cohort; it is reported, not drawn."""
    counts = region_counts(_sets(1000, 700, 400, any_n=720))
    assert counts['any'] == 720
    assert counts['history'] == 700


def test_text_outside_the_history_set_is_refused():
    """Containment is structural, so a breach means the ids and the arrays are misaligned."""
    sets = _sets(10, 3, 3)
    sets['text'] = set(range(5))
    with pytest.raises(ValueError, match='misaligned'):
        region_counts(sets)


def test_an_all_features_set_outside_a_single_feature_set_is_refused():
    """The reduction over every feature and the indexed predicates have to agree."""
    sets = _sets(1000, 700, 400, summary_n=300, diagnosis_n=250)
    sets['all_text'] = set(sets['text'])
    with pytest.raises(ValueError, match='misaligned'):
        region_counts(sets)


# ------------------------------------------------------------------------------------------
# The layout
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('all_n,history_n,text_n,summary_n,diagnosis_n', [
    (28600, 13649, 10400, 9878, 4920),   # the cohort's own proportions
    (1000, 999, 998, 998, 998),          # every ring thin, the features coincident
    (1000, 10, 1, 1, 1),                 # every ring wide
    (1000, 700, 400, 400, 10),           # one feature nearly all of the text set
    (1000, 700, 400, 200, 200),          # the two features disjoint
])
def test_every_circle_stays_inside_the_one_containing_its_superset(
        all_n, history_n, text_n, summary_n, diagnosis_n):
    counts = region_counts(_sets(all_n, history_n, text_n, summary_n, diagnosis_n))
    geometry = layout(counts)
    (x_sum, r_sum), (x_diag, r_diag) = geometry['summary'], geometry['diagnosis']
    reach = max(abs(x_sum) + r_sum, abs(x_diag) + r_diag)
    assert reach <= geometry['history_r'] + 1e-9, 'a text circle escapes the history circle'
    assert geometry['history_r'] <= geometry['all_r'] + 1e-12


def test_circle_areas_and_the_overlap_are_proportional_to_the_counts():
    """The claim the caption makes: the radii are square roots and the lens is solved."""
    counts = region_counts(_sets(28600, 13649, 10400, 9878, 4920))
    geometry = layout(counts)
    assert geometry['to_scale']

    unit = math.pi * geometry['all_r'] ** 2 / counts['all']
    (x_sum, r_sum), (x_diag, r_diag) = geometry['summary'], geometry['diagnosis']
    assert math.pi * geometry['history_r'] ** 2 == pytest.approx(
        unit * counts['history'], rel=1e-12)
    assert math.pi * r_sum ** 2 == pytest.approx(unit * counts['summary'], rel=1e-12)
    assert math.pi * r_diag ** 2 == pytest.approx(unit * counts['diagnosis'], rel=1e-12)

    overlap = lens_area(r_sum, r_diag, abs(x_diag - x_sum))
    assert overlap == pytest.approx(unit * counts['all_text'], rel=1e-6)


def test_the_cohort_proportions_do_not_need_the_squeeze():
    """At the real counts the pair fits, so the figure stays proportional and the caption does
    not have to disclaim it. Worth pinning: the discharge-summary set is most of the history
    set, which is the case that leaves least room to offset the pair."""
    geometry = layout(region_counts(_sets(28600, 13649, 10400, 9878, 4920)))
    assert geometry['to_scale']


def test_a_pair_that_cannot_fit_is_shrunk_and_reported():
    """A text set covering nearly all of its superset leaves no room to offset the pair."""
    counts = region_counts(_sets(28780, 23600, 23400, 12000, 11600))
    geometry = layout(counts)
    assert not geometry['to_scale']
    (x_sum, r_sum), (x_diag, r_diag) = geometry['summary'], geometry['diagnosis']
    reach = max(abs(x_sum) + r_sum, abs(x_diag) + r_diag)
    assert reach <= geometry['history_r'] + 1e-9, 'the squeeze did not bring the pair inside'


def test_an_empty_cohort_does_not_divide_by_zero():
    geometry = layout(region_counts(_sets(0, 0, 0)))
    assert geometry['history_r'] == 0.0
    assert geometry['summary'][1] == 0.0 and geometry['diagnosis'][1] == 0.0


def test_a_ring_can_be_too_thin_to_hold_its_count():
    """Recorded because the leader-line branch is otherwise exercised only by the real data,
    and a change to MIN_RING_LABEL_GAP would silently take a count off the figure."""
    geometry = layout(region_counts(_sets(28780, 23600, 9878, 9878, 4920)))
    assert geometry['all_r'] - geometry['history_r'] < MIN_RING_LABEL_GAP
