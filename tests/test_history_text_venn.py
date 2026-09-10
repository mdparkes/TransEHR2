"""Probes for the patient-count Euler diagram.

Two things here can be silently wrong. The containment of the text set inside the history set
is structural, not incidental: text history is built by intersecting the value stream's
observed history with the feature indicators, so a violation means the arrays and the id lists
are out of step and every count in the figure is untrustworthy. And the radii have to be
proportional to the counts by area, which is the claim the caption makes -- a figure whose
areas are wrong looks entirely plausible.

Concentric circles are what make the layout unconditional: nested counts give nested radii, so
there is nothing to solve and no arrangement in which a circle escapes its own superset. The
earlier layout placed two overlapping text circles inside the history circle, which needed an
area solver and a fit check because equal areas do not imply a containing arrangement. Those
probes went with the machinery.
"""

import math

import pytest

from plot_history_text_venn import MIN_RING_LABEL_GAP, layout, region_counts


def _sets(all_n, history_n, text_n, summary_n=0, diagnosis_n=0, any_n=None):
    """Nested patient id sets of the given sizes, with the text features inside the text set."""
    return {
        'all': set(range(all_n)),
        'readable': set(range(history_n)),
        'any': set(range(history_n if any_n is None else any_n)),
        'text': set(range(text_n)),
        'summary': set(range(summary_n)),
        'diagnosis': set(range(text_n - diagnosis_n, text_n)),
    }


def test_the_regions_partition_the_cohort():
    counts = region_counts(_sets(1000, 700, 400, summary_n=300, diagnosis_n=250))
    assert counts['text'] + counts['history_only'] == counts['history']
    assert counts['history'] + counts['no_history'] == counts['all']


def test_the_per_feature_counts_still_partition_the_text_set():
    """Reported rather than drawn, but the table has to add up."""
    counts = region_counts(_sets(1000, 700, 400, summary_n=300, diagnosis_n=250))
    assert counts['summary_only'] + counts['both'] == counts['summary']
    assert counts['diagnosis_only'] + counts['both'] == counts['diagnosis']


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


@pytest.mark.parametrize('all_n,history_n,text_n', [
    (28780, 23600, 9878),
    (1000, 999, 998),      # every ring vanishingly thin
    (1000, 10, 1),         # every ring wide
    (500, 500, 500),       # the three coincide
])
def test_every_circle_stays_inside_the_one_containing_its_superset(all_n, history_n, text_n):
    geometry = layout(region_counts(_sets(all_n, history_n, text_n)))
    assert geometry['text_r'] <= geometry['history_r'] + 1e-12
    assert geometry['history_r'] <= geometry['all_r'] + 1e-12


def test_circle_areas_are_proportional_to_the_counts():
    """The claim the caption makes, and the reason the radii are square roots."""
    counts = region_counts(_sets(28780, 23600, 9878))
    geometry = layout(counts)

    unit = math.pi * geometry['all_r'] ** 2 / counts['all']
    for key, radius_key in (('history', 'history_r'), ('text', 'text_r')):
        area = math.pi * geometry[radius_key] ** 2
        assert area == pytest.approx(unit * counts[key], rel=1e-12), (
            f'{key} circle area is not proportional to its count'
        )


def test_an_empty_cohort_does_not_divide_by_zero():
    geometry = layout(region_counts(_sets(0, 0, 0)))
    assert geometry['history_r'] == 0.0 and geometry['text_r'] == 0.0


def test_a_ring_can_be_too_thin_to_hold_its_count():
    """Which ring is thin depends on the cohort, and both cases occur.

    Recorded because the leader-line branch is otherwise exercised only by the real data, and
    a change to MIN_RING_LABEL_GAP would silently take a count off the figure. When most
    patients have history the outer ring is the thin one; at the current cohort proportions it
    is the inner ring, the text cohort filling most of the history circle.
    """
    outer_thin = layout(region_counts(_sets(28780, 23600, 9878)))
    assert outer_thin['all_r'] - outer_thin['history_r'] < MIN_RING_LABEL_GAP

    inner_thin = layout(region_counts(_sets(28600, 13649, 10400)))
    assert inner_thin['history_r'] - inner_thin['text_r'] < MIN_RING_LABEL_GAP
    assert inner_thin['all_r'] - inner_thin['history_r'] > MIN_RING_LABEL_GAP
