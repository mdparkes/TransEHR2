"""Probes for the patient-count Euler diagram.

Two things here can be silently wrong. The area solver returns a number for any input, so a
dropped factor produces a plausible-looking figure with the wrong overlap -- an earlier
revision lost a factor of pi exactly that way. And the containment of the text sets inside the
any-record set is structural, not incidental: text history is built by intersecting the value
stream's observed history with the feature indicator, so a violation means the arrays and the
id lists are out of step and every count in the figure is untrustworthy.
"""

import math

import pytest

from plot_history_text_venn import lens_area, region_counts, solve_distance


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
            'summary': set(range(5)),
            'diagnosis': set(),
        })
