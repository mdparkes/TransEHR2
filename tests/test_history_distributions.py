"""Probes for the historical record distribution figure.

Two ways this fails without failing loudly.

A cohort with no entry in `CAPTIONS` raises `KeyError` at the moment the figure is drawn, which
is after every partition has been read -- so the cost is the whole run, and the cohorts the
figure accepts are validated against `COHORTS` while the captions are a separate hand-written
map. They are held in step here instead.

The wording is the other. The figure is read on its own in a supplement, so its axis labels are
the only thing that says what a bar counts. "Historical" means records older than the extraction
boundary, which is a setting rather than a convention, and nothing on the figure gives it unless
the footer does.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_history_distributions as figure
from TransEHR2.data.cohorts import COHORTS


# ------------------------------------------------------------------------------------------
# Captions
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cohort', COHORTS)
def test_every_selectable_cohort_has_a_caption(cohort):
    """`--cohort` validates against COHORTS, so any name it accepts reaches the caption
    lookup -- and it is looked up after the arrays have been read."""
    assert cohort in figure.CAPTIONS, (
        f'--cohort {cohort} is accepted but has no caption, so the figure raises after the '
        f'whole extraction has been read'
    )


def test_no_caption_names_a_cohort_that_does_not_exist():
    assert set(figure.CAPTIONS) <= set(COHORTS)


@pytest.mark.parametrize('cohort,caption', sorted(figure.CAPTIONS.items()))
def test_a_caption_completes_the_sentence_it_is_substituted_into(cohort, caption):
    """The footer reads "n = 10,385 {caption}", so a caption starting with a capital or
    carrying its own count would read as two sentences."""
    assert caption.startswith('episodes')
    assert not caption.endswith('.')


# ------------------------------------------------------------------------------------------
# Wording
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cohort,caption', sorted(figure.CAPTIONS.items()))
def test_the_captions_say_historical(cohort, caption):
    """One term throughout the figure. The manuscript calls these records historical and the
    peri-stay ones peri-stay, and a figure that says pre-admission names neither."""
    assert 'historical' in caption
    assert 'pre-admission' not in caption.lower()


def test_the_axis_labels_say_historical_and_do_not_name_a_stream(tmp_path, monkeypatch):
    """Every event feature is also a valued feature, so naming the value stream on the count
    axis distinguishes it from nothing. And the labels are the only thing on the figure that
    says what a bar counts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = []
    real = plt.Axes.set_xlabel

    def record(self, text, *args, **kwargs):
        labels.append(text)
        return real(self, text, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, 'set_xlabel', record)
    figure.draw([('1', 5), ('2-4', 9)], [('0-48 h', 0), ('2-7 d', 10)],
                10385, figure.CAPTIONS['any_text'], str(tmp_path / 'figure.png'), '',
                cutoff_hours=48.0)

    assert labels == ['Historical records',
                      'Most recent historical record to ICU admission']
    assert not any('stream' in label for label in labels)


def test_the_footer_defines_historical_in_hours(tmp_path, monkeypatch):
    """48 hours is PREADMISSION_CUTOFF_HOURS, not a convention, so the figure has to say it."""
    captured = {}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    real_text = plt.Figure.text

    def record(self, x, y, text, *args, **kwargs):
        captured.setdefault('footers', []).append(text)
        return real_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, 'text', record)
    figure.draw([('1', 5)], [('0-6 h', 4)], 10385, figure.CAPTIONS['any_text'],
                str(tmp_path / 'figure.png'), '', cutoff_hours=48.0)

    footer = ' '.join(captured.get('footers', []))
    assert 'n = 10,385' in footer
    assert 'historical' in footer.lower()
    assert '48 hours before ICU admission' in footer


def test_the_footer_omits_the_definition_when_no_boundary_is_given(tmp_path, monkeypatch):
    """A cutoff of zero is the pre-boundary convention; claiming "more than 0 hours" would
    state something false rather than nothing."""
    captured = []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    real_text = plt.Figure.text

    def record(self, x, y, text, *args, **kwargs):
        captured.append(text)
        return real_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, 'text', record)
    figure.draw([('1', 5)], [('0-6 h', 4)], 100, figure.CAPTIONS['any_history'],
                str(tmp_path / 'figure.png'), '', cutoff_hours=0.0)

    footer = ' '.join(captured)
    assert 'n = 100' in footer
    assert 'hours before ICU admission' not in footer


# ------------------------------------------------------------------------------------------
# The gap axis
# ------------------------------------------------------------------------------------------

def test_the_gap_bins_partition_the_axis():
    """A gap falling in no bin is dropped silently; one falling in two is counted twice. main()
    checks the totals at runtime, which is after the arrays have been read."""
    bounds = [(lower, upper) for _, lower, upper in figure.GAP_BINS]
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == float('inf')
    for (_, upper), (lower, _) in zip(bounds, bounds[1:]):
        assert upper == lower, 'the bins leave a gap or overlap'


def test_the_first_gap_bin_spans_the_whole_peri_stay_era():
    """Every historical record is older than the extraction boundary, so this bin is empty by
    construction. It is kept as the check that it is: a bar here means a peri-stay record
    reached the history region. Splitting it finer spends bars on the same emptiness."""
    label, lower, upper = figure.GAP_BINS[0]
    assert (lower, upper) == (0.0, 48.0)
    assert label == '0-48 h'
    assert figure.GAP_BINS[1][0] == '2-7 d'


def test_nothing_lands_in_the_first_bin_when_the_cutoff_holds():
    """The property the bin exists to show, on gaps drawn from the far side of the boundary."""
    import numpy as np
    gaps = np.array([48.0, 72.0, 500.0, 10000.0, 90000.0])
    rows = figure.bin_counts(gaps, figure.GAP_BINS)
    assert rows[0] == ('0-48 h', 0)
    assert sum(count for _, count in rows) == gaps.size


def test_a_peri_stay_gap_is_visible_rather_than_dropped():
    """If the boundary were ever violated the figure has to show it, not silently omit it."""
    import numpy as np
    rows = figure.bin_counts(np.array([1.0, 47.0, 72.0]), figure.GAP_BINS)
    assert rows[0] == ('0-48 h', 2)
