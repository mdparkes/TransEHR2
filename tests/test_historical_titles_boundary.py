"""Probes for which source records the diagnosis audit calls historical.

`historical_titles` reconciles two sources: `n_retained` comes from the extracted arrays and
says how many pre-admission diagnosis records survived, and the titles come from the episode
timeseries CSV. The reconciliation is a suffix -- extraction keeps the most recent
pre-admission records -- and it is only valid when both sides cut the era at the same place.

Cut the CSV at admission while the arrays cut at the pre-admission cutoff and the failure is
silent: the suffix is taken from a longer run, so the function returns the peri-stay records
the model never reads, and the record count still agrees. Two manuscript tables are built on
this, so the boundary is pinned rather than trusted.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit_historic_diagnoses import TEXT_FEATURE, historical_titles


# Two earlier admissions and one work-up inside the peri-stay window. Under a 48 h cutoff only
# the first two are historical; the -30 h record belongs to the stay being predicted.
RECORDS = [
    (-5000.0, 'Chronic kidney disease'),
    (-800.0, 'Congestive heart failure'),
    (-30.0, 'Acute respiratory failure'),
]


@pytest.fixture
def episode(tmp_path):
    """Write an episodeN.csv and its timeseries, returning the episode path."""
    frame = pd.DataFrame({'Hours': [hour for hour, _ in RECORDS],
                          TEXT_FEATURE: [title for _, title in RECORDS]})
    frame.to_csv(tmp_path / 'episode1_timeseries.csv', index=False)
    (tmp_path / 'episode1.csv').write_text('Icustay\n1\n')
    return str(tmp_path / 'episode1.csv')


def test_the_cutoff_excludes_the_peri_stay_record(episode):
    """With one record retained, it must be the most recent record before the cutoff."""
    titles, n_records, n_in_source, status = historical_titles(
        episode, n_retained=1, preadmission_cutoff_hours=48.0)
    assert status == 'ok'
    assert titles == ['Congestive heart failure']
    assert n_records == 1
    # The peri-stay record is not in the source population either.
    assert n_in_source == 2


def test_cutting_at_admission_returns_the_wrong_record(episode):
    """The failure the cutoff prevents, stated so the guard cannot be removed quietly.

    At a cutoff of 0 the source run is three records long and its one-record suffix is the
    peri-stay work-up -- text the model does not read -- while `n_retained` still says one.
    """
    titles, n_records, _, status = historical_titles(
        episode, n_retained=1, preadmission_cutoff_hours=0.0)
    assert status == 'ok'
    assert titles == ['Acute respiratory failure']
    assert n_records == 1


def test_all_pre_cutoff_records_are_returned_when_all_were_retained(episode):
    titles, n_records, n_in_source, status = historical_titles(
        episode, n_retained=2, preadmission_cutoff_hours=48.0)
    assert status == 'ok'
    assert titles == ['Chronic kidney disease', 'Congestive heart failure']
    assert n_records == 2 and n_in_source == 2


def test_retaining_more_than_the_source_holds_is_reported(episode):
    """The arrays and the CSV disagreeing about the episode must not pass as data."""
    _, _, n_in_source, status = historical_titles(
        episode, n_retained=3, preadmission_cutoff_hours=48.0)
    assert status == 'fewer_records_than_arrays'
    assert n_in_source == 2


def test_retaining_nothing_returns_nothing(episode):
    titles, n_records, n_in_source, status = historical_titles(
        episode, n_retained=0, preadmission_cutoff_hours=48.0)
    assert status == 'ok'
    assert titles == [] and n_records == 0
    assert n_in_source == 2


def test_pipe_delimited_titles_are_split(tmp_path):
    frame = pd.DataFrame({'Hours': [-900.0],
                          TEXT_FEATURE: ['Sepsis|Acute kidney injury']})
    frame.to_csv(tmp_path / 'episode1_timeseries.csv', index=False)
    titles, n_records, _, status = historical_titles(
        str(tmp_path / 'episode1.csv'), n_retained=1, preadmission_cutoff_hours=48.0)
    assert status == 'ok'
    assert titles == ['Sepsis', 'Acute kidney injury']
    assert n_records == 1
