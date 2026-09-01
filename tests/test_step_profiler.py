"""Probes for the training-step profiler.

The profiler exists to answer where a pretraining step's wall clock goes, and its numbers are
read straight off the batch script's output. What has to hold is that it discards its warmup,
attributes intervals to the phase named at the closing mark, and stops the loop on schedule.
"""

import time

import pytest

from TransEHR2.utils import NullStepProfiler, StepProfiler


def test_null_profiler_never_stops_the_loop():
    """The production path must run the epoch to its end."""
    profiler = NullStepProfiler()
    assert profiler.enabled is False
    for _ in range(50):
        profiler.mark('forward')
        assert profiler.end_step() is False


def test_end_step_stops_the_loop_at_total_steps():
    profiler = StepProfiler(total_steps=6, warmup=2)
    stops = [profiler.end_step() for _ in range(6)]
    assert stops == [False] * 5 + [True]


def test_total_steps_cannot_fall_below_warmup():
    """A total at or under the warmup would measure nothing and report an empty table."""
    profiler = StepProfiler(total_steps=1, warmup=3)
    assert profiler.total_steps == 4


def test_warmup_steps_are_discarded():
    profiler = StepProfiler(total_steps=5, warmup=3)
    for _ in range(5):
        profiler.mark('forward')
        profiler.end_step()
    # Steps 3 and 4 are past warmup; each contributes one 'forward' interval.
    assert len(profiler.timings['forward']) == 2
    assert len(profiler.timings['TOTAL']) == 2


def test_an_interval_is_attributed_to_the_phase_at_its_closing_mark():
    """`mark` names the work that just finished, not the work about to start."""
    profiler = StepProfiler(total_steps=2, warmup=0)
    profiler.mark('first')          # opens the step; nothing to attribute yet
    time.sleep(0.05)
    profiler.mark('slow phase')
    profiler.mark('fast phase')
    profiler.end_step()
    assert profiler.timings['slow phase'][0] >= 0.05
    assert profiler.timings['fast phase'][0] < 0.05


def test_report_survives_a_run_that_never_passed_warmup(capsys):
    """A job killed early must print a note rather than raise out of the training loop."""
    profiler = StepProfiler(total_steps=5, warmup=3)
    profiler.mark('forward')
    profiler.report()
    assert 'no steps completed past warmup' in capsys.readouterr().out


def test_report_extrapolates_with_the_step_count_it_is_given(capsys):
    """The truncated loader's length would understate the epoch by the truncation factor."""
    profiler = StepProfiler(total_steps=1, warmup=0)
    profiler.mark('forward')
    profiler.end_step()
    profiler.report(steps_per_epoch=96)
    out = capsys.readouterr().out
    assert '96 training steps per full epoch' in out
    assert 'implied training time per epoch' in out
