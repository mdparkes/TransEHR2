"""Probes for the historical record distribution figure.

Three ways this fails without failing loudly.

A cohort with no entry in `CAPTIONS` raises `KeyError` at the moment the figure is drawn, which
is after every partition has been read -- so the cost is the whole run, and the cohorts the
figure accepts are validated against `COHORTS` while the captions are a separate hand-written
map. They are held in step here instead.

The wording is the second. The figure is read on its own in a supplement, so its axis labels are
the only thing that says what a bar counts. "Historical" means records older than the extraction
boundary, which is a setting rather than a convention, and nothing on the figure gives it unless
the footer does.

The interval panel is the third, and it is the one that reads as wrong only if you already know
the answer. Its bars count intervals where the other two count episodes, and the intervals are
selected out of a pooled array by the episode each came from -- so a cohort filter that missed
the renumbering, or a differencing that crossed from one episode into the next, would produce a
panel of the right shape carrying the wrong population.
"""

import os
import sys

import numpy as np
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
    figure.draw([('1', 5), ('2-4', 9)], [('<1 h', 3), ('>1 y', 7)],
                [('0-48 h', 0), ('2-7 d', 10)],
                10385, figure.CAPTIONS['any_text'], str(tmp_path / 'figure.png'), '',
                cutoff_hours=48.0)

    assert labels == ['Historical records',
                      'Between consecutive historical records',
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
    figure.draw([('1', 5)], [('<1 h', 4)], [('0-6 h', 4)], 10385,
                figure.CAPTIONS['any_text'],
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
    figure.draw([('1', 5)], [('<1 h', 4)], [('0-6 h', 4)], 100,
                figure.CAPTIONS['any_history'],
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
    gaps = np.array([48.0, 72.0, 500.0, 10000.0, 90000.0])
    rows = figure.bin_counts(gaps, figure.GAP_BINS)
    assert rows[0] == ('0-48 h', 0)
    assert sum(count for _, count in rows) == gaps.size


def test_a_peri_stay_gap_is_visible_rather_than_dropped():
    """If the boundary were ever violated the figure has to show it, not silently omit it."""
    rows = figure.bin_counts(np.array([1.0, 47.0, 72.0]), figure.GAP_BINS)
    assert rows[0] == ('0-48 h', 2)


# ------------------------------------------------------------------------------------------
# The interval panel
# ------------------------------------------------------------------------------------------

HIST = 6
EPISODE = 3

# Three episodes, in hours relative to ICU admission. The second holds a single record and so
# contributes no interval; the third holds two records half an hour apart, which is what a
# single prior admission looks like next to the years between admissions.
RECORDS = [
    [-1000.0, -100.0, -60.0],
    [-500.0],
    [-10000.0, -9000.0, -200.0, -199.5],
]

# What differencing across the era boundary would produce, per episode: the distance from the
# last historical record to the first peri-stay one. None of these is a real interval, and the
# fixture is chosen so none of them collides with one.
BOUNDARY_CROSSINGS = [60.0, 500.0, 199.5]


def history(records, reverse_columns=False):
    """(times, masks) with each episode's history right-justified in [0, HIST).

    The peri-stay region is filled as well, at t = 0, 1, 2. It is what the interval computation
    must not reach into, so leaving it empty would make the probe pass for the wrong reason.
    """
    times = np.zeros((len(records), HIST + EPISODE))
    masks = np.zeros((len(records), HIST + EPISODE))
    for row, stamps in enumerate(records):
        stamps = list(stamps)
        start = HIST - len(stamps)
        times[row, start:HIST] = stamps[::-1] if reverse_columns else stamps
        masks[row, start:HIST] = 1
        times[row, HIST:] = np.arange(EPISODE, dtype=float)
        masks[row, HIST:] = 1
    return times, masks


def test_an_episode_contributes_one_interval_per_consecutive_pair():
    """k records give k - 1 intervals, which is what makes the panel's denominator differ from
    the other two panels' rather than merely being larger."""
    intervals, _, _ = figure.history_intervals(*history(RECORDS), HIST)
    assert len(intervals) == sum(max(len(stamps) - 1, 0) for stamps in RECORDS)


def test_the_intervals_are_the_differences_between_consecutive_records():
    intervals, _, _ = figure.history_intervals(*history(RECORDS), HIST)
    assert sorted(intervals) == [0.5, 40.0, 900.0, 1000.0, 8800.0]


def test_no_interval_crosses_into_the_peri_stay_region():
    """The two eras are adjacent columns of one array. A difference taken across the boundary
    would be the gap the right-hand panel reports, entering the middle panel as though it were
    time between two historical records."""
    intervals, _, _ = figure.history_intervals(*history(RECORDS), HIST)
    for crossing in BOUNDARY_CROSSINGS:
        assert crossing not in set(intervals)


def test_an_episode_holding_one_record_contributes_no_interval_and_no_median():
    intervals, episode, median = figure.history_intervals(*history(RECORDS), HIST)
    assert 1 not in set(episode)
    assert np.isnan(median[1])


def test_each_interval_carries_the_episode_it_came_from():
    """The cohort filter selects intervals through this index, so an index that did not track
    the record it was differenced from would hand the panel another episode's history."""
    intervals, episode, _ = figure.history_intervals(*history(RECORDS), HIST)
    assert list(episode) == [0, 0, 2, 2, 2]
    assert sorted(intervals[episode == 0]) == [40.0, 900.0]
    assert sorted(intervals[episode == 2]) == [0.5, 1000.0, 8800.0]


def test_the_per_episode_median_is_taken_within_the_episode():
    """The figure pools intervals, which weights an episode by how many records it holds. The
    report gives this alongside it, and it is a different number."""
    _, _, median = figure.history_intervals(*history(RECORDS), HIST)
    assert median[0] == 470.0
    assert median[2] == 1000.0


def test_the_column_order_does_not_change_the_intervals():
    """Records are ordered by timestamp before differencing rather than taken in column order,
    so the panel is a property of the timestamps and not of the array layout."""
    forward, masks = history(RECORDS)
    backward, _ = history(RECORDS, reverse_columns=True)
    ascending, _, _ = figure.history_intervals(forward, masks, HIST)
    descending, _, _ = figure.history_intervals(backward, masks, HIST)
    assert sorted(ascending) == sorted(descending)


def test_no_interval_is_negative():
    """A negative interval would bin below the axis and vanish. Sorting is what rules it out,
    so this holds however the columns are written."""
    for reverse in (False, True):
        intervals, _, _ = figure.history_intervals(
            *history(RECORDS, reverse_columns=reverse), HIST)
        assert intervals.min() >= 0.0


def test_an_empty_history_region_yields_nothing_rather_than_raising():
    times, masks = history(RECORDS)
    masks[:, :HIST] = 0
    intervals, episode, median = figure.history_intervals(times, masks, HIST)
    assert intervals.size == 0 and episode.size == 0
    assert np.isnan(median).all()


# ------------------------------------------------------------------------------------------
# The interval axis
# ------------------------------------------------------------------------------------------

def test_the_interval_bins_partition_the_axis():
    bounds = [(lower, upper) for _, lower, upper in figure.INTERVAL_BINS]
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == float('inf')
    for (_, upper), (lower, _) in zip(bounds, bounds[1:]):
        assert upper == lower, 'the bins leave a gap or overlap'


def test_two_records_sharing_a_timestamp_land_in_the_first_bin():
    """An interval of exactly zero says two records were recorded at the same time, which is a
    fact about the history rather than a value to drop on the floor."""
    rows = figure.bin_counts(np.array([0.0, 0.5, 2.0]), figure.INTERVAL_BINS)
    assert rows[0] == ('<1 h', 2)
    assert sum(count for _, count in rows) == 3


def test_the_interval_bins_reach_sub_day_resolution():
    """Consecutive records within one prior admission sit hours apart. Binning them with the
    gap panel's edges would put every one of them in a single bar."""
    edges = [upper for _, _, upper in figure.INTERVAL_BINS]
    assert sum(1 for edge in edges if edge <= 24.0) >= 3


# ------------------------------------------------------------------------------------------
# What the panel says it counts
# ------------------------------------------------------------------------------------------

def test_the_middle_panel_counts_intervals_and_the_others_count_episodes(tmp_path, monkeypatch):
    """Two units on one figure. The y labels are the only thing separating them, so a reader
    who takes the middle panel for a count of episodes gets a population that does not exist."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = []
    real = plt.Axes.set_ylabel

    def record(self, text, *args, **kwargs):
        labels.append(text)
        return real(self, text, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, 'set_ylabel', record)
    figure.draw([('1', 5)], [('<1 h', 4)], [('0-48 h', 0)], 10385,
                figure.CAPTIONS['any_text'], str(tmp_path / 'figure.png'), '',
                cutoff_hours=48.0)

    assert labels == ['Episodes', 'Intervals', 'Episodes']


def test_the_footer_states_the_middle_panel_denominator(tmp_path, monkeypatch):
    """The y label names the unit; the footer is what says why the middle panel's total is not
    the n the other two are drawn against."""
    captured = []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    real_text = plt.Figure.text

    def record(self, x, y, text, *args, **kwargs):
        captured.append(text)
        return real_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, 'text', record)
    figure.draw([('1', 5)], [('<1 h', 4)], [('0-48 h', 0)], 10385,
                figure.CAPTIONS['any_text'], str(tmp_path / 'figure.png'), '',
                cutoff_hours=48.0)

    footer = ' '.join(captured)
    assert 'intervals rather than episodes' in footer
    assert 'one fewer interval than it holds records' in footer


# ------------------------------------------------------------------------------------------
# Pooling the partitions
# ------------------------------------------------------------------------------------------

def partition(patients, counts, intervals, interval_episode):
    """One collect_partition return, with everything the merge touches."""
    n = len(patients)
    return {
        'patient': np.asarray(patients),
        'val_count': np.asarray(counts),
        'val_latest': np.full(n, -100.0),
        'event_latest': np.full(n, -100.0),
        'median_interval': np.full(n, 1.0),
        'in_cohort': np.ones(n, dtype=bool),
        'in_value': np.ones(n, dtype=bool),
        'in_any': np.ones(n, dtype=bool),
        'intervals': np.asarray(intervals, dtype=float),
        'interval_episode': np.asarray(interval_episode),
        'hist': HIST,
    }


def test_the_interval_index_is_renumbered_when_partitions_are_concatenated(monkeypatch):
    """Each partition indexes its own rows. Concatenating the per-episode arrays renumbers
    those rows, and an index left at the partition's own numbering would still be in range --
    so the cohort filter would select real intervals belonging to the wrong episodes."""
    parts = [
        partition([1, 2], [2, 3], [10.0, 20.0, 30.0], [0, 1, 1]),
        partition([3], [4], [40.0, 50.0, 60.0], [0, 0, 0]),
    ]
    supply = iter(parts)
    monkeypatch.setattr(figure, 'collect_partition',
                        lambda *args, **kwargs: next(supply))
    monkeypatch.setattr(figure.os.path, 'isdir', lambda path: True)

    merged = figure.collect('data', ['fold0'], ['train', 'test'], 'any_text')

    assert list(merged['interval_episode']) == [0, 1, 1, 2, 2, 2]
    assert list(merged['intervals']) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert len(merged['val_count']) == 3
