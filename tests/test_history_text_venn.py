"""Probes for the patient-count Euler diagram.

Three things here can be silently wrong. The area solver returns a number for any input, so a
dropped factor produces a plausible-looking figure with the wrong overlap -- an earlier
revision lost a factor of pi exactly that way. The containment of the text sets inside the
any-record set is structural, not incidental: text history is built by intersecting the value
stream's observed history with the feature indicator, so a violation means the arrays and the
id lists are out of step and every count in the figure is untrustworthy. And the layout has to
keep every circle inside its superset, which area proportionality alone does not guarantee: a
pair of equal-area circles can still be arranged so that one crosses the boundary of the
circle that contains its set.
"""

import math

import pytest

from plot_history_text_venn import (MIN_RING_LABEL_GAP, layout, lens_area, region_counts,
                                    solve_distance)


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


def test_the_regions_partition_each_set():
    counts = region_counts({
        'all': set(range(1000)),
        'any': set(range(700)),
        'readable': set(range(650)),
        'summary': set(range(400)),
        'diagnosis': set(range(250, 600)),
    })
    assert counts['summary_only'] + counts['both'] == counts['summary']
    assert counts['diagnosis_only'] + counts['both'] == counts['diagnosis']
    assert (counts['summary_only'] + counts['diagnosis_only'] + counts['both']
            + counts['any_only']) == counts['any']
    assert counts['any'] + counts['no_history'] == counts['all']


def test_text_outside_the_any_record_set_is_refused():
    """Containment is structural, so a breach means the ids and the arrays are misaligned."""
    with pytest.raises(ValueError, match='misaligned'):
        region_counts({
            'all': set(range(10)),
            'any': set(range(3)),
            'readable': set(range(3)),
            'summary': set(range(5)),
            'diagnosis': set(),
        })


def _counts(n_all, n_any, n_sum, n_diag, n_both):
    """Region counts for a nested arrangement of the requested sizes."""
    return {
        'all': n_all, 'any': n_any, 'summary': n_sum, 'diagnosis': n_diag, 'both': n_both,
        'summary_only': n_sum - n_both, 'diagnosis_only': n_diag - n_both,
        'any_only': n_any - n_sum - n_diag + n_both, 'no_history': n_all - n_any,
    }


@pytest.mark.parametrize('counts', [
    # The cohort's own shape: most patients carry history, and the text sets overlap heavily.
    _counts(28780, 23600, 9878, 12000, 8500),
    # Two large, nearly disjoint text sets that cannot fit inside the history circle at scale.
    _counts(28780, 23600, 12000, 11500, 200),
    # Every patient carries history, collapsing the outer ring to nothing.
    _counts(23600, 23600, 9878, 12000, 8500),
    # One text set only.
    _counts(28780, 23600, 9878, 0, 0),
])
def test_every_circle_stays_inside_the_one_containing_its_superset(counts):
    """Equal areas do not imply a containing arrangement; the layout has to enforce it."""
    geometry = layout(counts)
    assert geometry['any_r'] <= geometry['all_r'] + 1e-9
    for key in ('summary', 'diagnosis'):
        centre, radius = geometry[key]
        assert abs(centre) + radius <= geometry['any_r'] + 1e-9, key


def test_circle_areas_are_proportional_to_the_counts():
    counts = _counts(28780, 23600, 9878, 12000, 8500)
    geometry = layout(counts)
    assert geometry['to_scale']
    # The cohort circle has radius 1, so a circle's area over pi is its share of the cohort.
    for key, n in (('summary', counts['summary']), ('diagnosis', counts['diagnosis'])):
        assert geometry[key][1] ** 2 == pytest.approx(n / counts['all'])
    assert geometry['any_r'] ** 2 == pytest.approx(counts['any'] / counts['all'])
    (x_sum, r_sum), (x_diag, r_diag) = geometry['summary'], geometry['diagnosis']
    overlap = lens_area(r_sum, r_diag, abs(x_diag - x_sum))
    assert overlap / math.pi == pytest.approx(counts['both'] / counts['all'])


def test_a_pair_that_cannot_fit_is_shrunk_and_reported():
    """The figure has to say when it stopped being to scale, or it misrepresents the counts."""
    geometry = layout(_counts(28780, 23600, 12000, 11500, 200))
    assert not geometry['to_scale']
    assert geometry['summary'][1] ** 2 < 12000 / 28780


def test_the_ring_between_the_cohort_and_history_circles_is_thin_for_this_cohort():
    """The no-history ring holds 18% of patients but little of the radius, so its count goes
    on a leader line. Pinning that keeps the label from being placed inside a ring too thin to
    hold it."""
    geometry = layout(_counts(28780, 23600, 9878, 12000, 8500))
    assert geometry['all_r'] - geometry['any_r'] < MIN_RING_LABEL_GAP
