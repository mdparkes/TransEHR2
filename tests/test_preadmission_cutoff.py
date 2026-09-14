"""Probes for the pre-admission cutoff that separates the two eras.

`PREADMISSION_CUTOFF_HOURS` moves the boundary between the history region and the episode
region away from ICU admission. Two things have to hold and neither raises when it does not:

* records between the cutoff and admission must leave the history region and enter the episode
  region, because the history region is what the cohort predicates and the record switches act
  on positionally;
* the minimum in-stay timestep count must stay anchored at admission, because it exists to
  require ICU data and a run of pre-admission records inside the episode region is not that.

A cutoff of 0 must reproduce the old behaviour exactly, which is what makes the parameter safe
to leave at its extraction-time value and vary at load time.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransEHR2.data.preprocessing import (_get_tensor_dimensions,
                                          _init_tensorized_worker,
                                          _process_single_episode,
                                          filter_timeseries_records)


def frame(hours, column='Heart rate'):
    """A one-column frame indexed by hours relative to admission."""
    return pd.DataFrame({column: np.arange(len(hours), dtype=float)},
                        index=pd.to_timedelta(list(hours), unit='h'))


def split(hours, cutoff=0.0, max_history=500, max_episode=100, max_hours=48):
    """Return (retained hours, history length) for one stream under a cutoff."""
    values = frame(hours)
    kept, _, _, history_len, _, _ = filter_timeseries_records(
        values, frame(hours, 'Lactate'), None, max_history, max_episode, max_hours, cutoff)
    return [h / np.timedelta64(1, 'h') for h in kept.index.to_numpy()], history_len


HOURS = [-2000.0, -100.0, -30.0, -10.0, -1.0, 0.0, 5.0, 40.0]


def test_a_zero_cutoff_splits_at_admission():
    """The default has to be the behaviour that predates the parameter."""
    kept, history_len = split(HOURS, cutoff=0.0)
    assert kept == HOURS
    # Everything strictly before admission is history: five records.
    assert history_len == 5


def test_the_cutoff_moves_records_out_of_the_history_region():
    kept, history_len = split(HOURS, cutoff=48.0)
    assert kept == HOURS
    # Only -2000 and -100 are earlier than 48 h before admission.
    assert history_len == 2


def test_a_record_at_the_boundary_belongs_to_the_episode_region():
    """The episode region is closed at the cutoff, matching the manuscript's closed interval."""
    _, history_len = split([-60.0, -48.0, 10.0], cutoff=48.0)
    assert history_len == 1


def test_the_history_cap_keeps_the_most_recent_pre_cutoff_records():
    kept, history_len = split([-500.0, -400.0, -300.0, -60.0, 1.0], cutoff=48.0,
                              max_history=2)
    assert history_len == 2
    assert kept == [-300.0, -60.0, 1.0]


def test_the_episode_cap_drops_the_stay_before_the_pre_admission_run():
    """The hazard the config comment warns about, pinned so the warning cannot go stale.

    Truncation keeps the earliest records of the episode region. With a cutoff those are
    pre-admission, so a cap smaller than the region holds discards the ICU stay and keeps the
    run before it -- silently, since the episode still passes every length check.
    """
    kept, history_len = split([-40.0, -30.0, -20.0, 1.0, 2.0], cutoff=48.0, max_episode=3)
    assert history_len == 0
    assert kept == [-40.0, -30.0, -20.0], 'the ICU stay survived a cap it should not have'
    assert 1.0 not in kept and 2.0 not in kept


def test_the_episode_window_is_still_bounded_above():
    kept, _ = split([-60.0, -10.0, 10.0, 47.0, 48.0, 100.0], cutoff=48.0, max_hours=48)
    assert 48.0 not in kept and 100.0 not in kept
    assert kept == [-60.0, -10.0, 10.0, 47.0]


def test_the_two_streams_are_split_at_the_same_boundary():
    """The value and event streams are filtered separately and must agree on the era."""
    values = frame([-60.0, -30.0, 5.0])
    events = frame([-60.0, -30.0, 5.0], 'Lactate')
    _, _, _, val_history, event_history, _ = filter_timeseries_records(
        values, events, None, 500, 100, 48, 48.0)
    assert val_history == event_history == 1


# ------------------------------------------------------------------------------------------
# The in-stay minimum, through the episode processor
# ------------------------------------------------------------------------------------------

VARIABLE_PROPERTIES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
    'variable_properties.yaml')

VALUED, EVENT, STATIC = ['Heart rate'], ['Lactate'], ['Age', 'Gender']
HISTORY_STEPS, EPISODE_STEPS = 20, 100


class _Reader:
    """The slice of MIMICDataReader's interface that `_process_single_episode` uses."""

    def __init__(self, hours):
        self.hours = list(hours)

    def __getitem__(self, index):
        stamps = pd.to_timedelta(self.hours, unit='h')
        values = pd.DataFrame({'Heart rate': np.arange(len(self.hours), dtype=float)},
                              index=stamps)
        events = pd.DataFrame({'Lactate': np.ones(len(self.hours))}, index=stamps)
        statics = pd.Series({'Age': 60.0, 'Gender': 'M'})
        return None, statics, values, events, None, (0.0, 120.0, np.zeros(3, dtype=np.float32))


@pytest.fixture
def processor():
    """Initialize the module-level worker state `_process_single_episode` reads."""
    dims = _get_tensor_dimensions(VARIABLE_PROPERTIES, VALUED, EVENT, [], STATIC,
                                  max_ts_len=HISTORY_STEPS + EPISODE_STEPS, n_episodes=1,
                                  phenotype_dim=3)
    _init_tensorized_worker(
        VARIABLE_PROPERTIES, VALUED, EVENT, [], STATIC,
        {name: getattr(dims, name) for name in dims.__dataclass_fields__})


def process(hours, cutoff, min_steps=10):
    """Run one episode through the processor, returning None when it is filtered out."""
    return _process_single_episode(0, _Reader(hours), HISTORY_STEPS, EPISODE_STEPS, 48,
                                   min_steps, 48, cutoff)


# Twenty hours sitting between the cutoff and admission, so they land in the episode region
# without being ICU data.
PERI_STAY = [-40.0 + offset for offset in range(20)]


def test_peri_stay_records_cannot_satisfy_the_in_stay_minimum(processor):
    nine = [float(hour) for hour in range(9)]
    assert process(PERI_STAY + nine, cutoff=48.0) is None, (
        'twenty peri-stay timesteps stood in for the ten in-stay timesteps the minimum asks '
        'for'
    )


def test_the_minimum_is_met_by_in_stay_records_alone(processor):
    ten = [float(hour) for hour in range(10)]
    episode = process(PERI_STAY + ten, cutoff=48.0)
    assert episode is not None
    # Every retained timestep is in the episode region, none in history.
    assert episode.val_history_len == 0
    assert episode.val_len == len(PERI_STAY) + len(ten)


def test_the_minimum_is_unchanged_at_a_zero_cutoff(processor):
    """Without a cutoff the peri-stay run is history, and the minimum behaves as before."""
    nine = [float(hour) for hour in range(9)]
    assert process(PERI_STAY + nine, cutoff=0.0) is None
    ten = [float(hour) for hour in range(10)]
    episode = process(PERI_STAY + ten, cutoff=0.0)
    assert episode is not None
    assert episode.val_history_len == len(PERI_STAY)


@pytest.mark.parametrize('cutoff', [0.0, 24.0, 48.0])
def test_history_and_episode_lengths_account_for_every_retained_record(cutoff):
    values = frame(HOURS)
    kept, _, _, history_len, _, _ = filter_timeseries_records(
        values, frame(HOURS, 'Lactate'), None, 500, 100, 48, cutoff)
    assert 0 <= history_len <= len(kept)
    boundary = -pd.Timedelta(hours=cutoff)
    assert (kept.index[:history_len] < boundary).all()
    assert (kept.index[history_len:] >= boundary).all()
