"""Probes for the training-step profiler.

The profiler exists to answer where a pretraining step's wall clock goes, and its numbers are
read straight off the batch script's output. What has to hold is that it discards its warmup,
attributes intervals to the phase named at the closing mark, and stops the loop on schedule.
"""

import argparse
import ast
import inspect
import os
import sys
import textwrap
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_phase2_pipeline
from test_phase2_pipeline import EPOCH_LINE, STARTUP_LINE
from TransEHR2.utils import EPOCH_TIMING_PREFIX, STARTUP_TIMING_PREFIX
from TransEHR2.utils import NullStepProfiler, StepProfiler, report_epoch_timing


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


class TestEpochTimingReport:
    """`report_epoch_timing` prints the line the smoke test sizes the sweep from."""

    def test_the_first_epoch_is_excluded_from_the_mean(self, capsys):
        """It carries worker spawn, cuDNN autotuning and allocator growth, none of which recur."""
        report_epoch_timing([100.0, 10.0, 12.0], total_epoch=200)
        out = capsys.readouterr().out
        assert 'mean=11.0000' in out
        assert 'first=100.0000' in out

    def test_a_single_epoch_falls_back_to_itself(self, capsys):
        report_epoch_timing([12.5], total_epoch=200)
        out = capsys.readouterr().out
        assert 'mean=12.5000' in out
        assert 'only one epoch ran' in out

    def test_no_epochs_still_emits_a_parseable_line(self, capsys):
        """A killed or resumed-and-finished run must not leave the parser matching nothing."""
        report_epoch_timing([], total_epoch=200)
        assert capsys.readouterr().out.strip() == f'{EPOCH_TIMING_PREFIX} n=0'

    def test_the_smoke_test_regexes_match_what_is_printed(self, capsys):
        """The two modules are coupled by a text format, so pin it from both ends."""
        report_epoch_timing([100.0, 10.0, 12.0], total_epoch=200)
        out = capsys.readouterr().out
        match = EPOCH_LINE.search(out)
        assert match is not None
        assert int(match.group(1)) == 3
        assert float(match.group(2)) == pytest.approx(11.0)
        assert float(match.group(3)) == pytest.approx(100.0)

    def test_the_startup_regex_matches_the_line_run_experiment_prints(self):
        line = f'\n{STARTUP_TIMING_PREFIX} 28.44\n'
        match = STARTUP_LINE.search(line)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(28.44)


class TestTimingIsMeasuredNotExtrapolated:
    """The old estimator divided whole-process wall time by the epoch count, which multiplied
    fixed startup by both the epoch budget and the episode ratio -- a factor of ~4,800 that
    reported a 0.6 h trial as 651 h. The timing stage must not reintroduce that."""

    def test_the_timing_stage_does_not_truncate_its_episodes(self):
        # Docstring stripped first: it names the flag to explain the absence, and matching that
        # would keep this probe green whatever the code did.
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(test_phase2_pipeline.run_stage_timing)))
        function = tree.body[0]
        if (isinstance(function.body[0], ast.Expr)
                and isinstance(function.body[0].value, ast.Constant)):
            function.body.pop(0)
        literals = {node.value for node in ast.walk(function)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert '--limit_episodes' not in literals

    def test_report_timing_prices_from_a_per_epoch_measurement(self):
        source = inspect.getsource(test_phase2_pipeline.report_timing)
        # No episode-ratio scaling anywhere: the measurement is already at full size.
        assert 'n_train' not in source
        assert "entry['epoch']" in source or 'entry["epoch"]' in source

    def test_report_timing_recommends_from_the_slower_arm(self):
        """One --time covers the whole array, so the faster arm cannot set it."""
        timings = {
            'additive': {'startup': 30.0, 'epoch': 10.0, 'epochs': 3, 'first_epoch': 40.0,
                         'wall': 100.0},
            'rope': {'startup': 30.0, 'epoch': 20.0, 'epochs': 3, 'first_epoch': 50.0,
                     'wall': 130.0},
        }
        args = argparse.Namespace(arms=['additive', 'rope'])
        report = test_phase2_pipeline.Report()
        test_phase2_pipeline.report_timing(
            args, report, timings, {'PRETRAIN_TOTAL_EPOCH': 200, 'FINETUNE_TOTAL_EPOCH': 500}
        )
        # rope: 30 + 200*20 = 4030 s = 1.12 h; x1.5 -> 1.68 h -> 2 h.
        assert any('pretrain 02:00:00' in note for note in report.notes)
        assert any('rope' in note for note in report.notes)

    def test_report_timing_says_nothing_when_no_arm_was_measured(self):
        report = test_phase2_pipeline.Report()
        test_phase2_pipeline.report_timing(
            argparse.Namespace(arms=['additive']), report, {}, {'PRETRAIN_TOTAL_EPOCH': 200}
        )
        assert any('No --time recommendation' in note for note in report.notes)


class TestTimingStageRerunHazards:
    """A rerun has to produce fresh numbers or say plainly that it did not."""

    def test_a_zero_epoch_line_is_recognised_as_a_resumed_run(self):
        """`pretrain_model` resuming at its epoch budget completes nothing and measures nothing.
        Reported as 'no line printed' this would look like a format problem, not a stale
        checkpoint."""
        assert test_phase2_pipeline.ZERO_EPOCH_LINE.search(
            f'\n{EPOCH_TIMING_PREFIX} n=0\n') is not None
        assert test_phase2_pipeline.ZERO_EPOCH_LINE.search(
            f'\n{EPOCH_TIMING_PREFIX} n=3 mean=17.5000 first=48.0 steady_n=2\n') is None

    def test_a_zero_epoch_line_does_not_parse_as_a_measurement(self):
        """EPOCH_LINE must not match it: n=0 carries no mean to price the request from."""
        assert EPOCH_LINE.search(f'{EPOCH_TIMING_PREFIX} n=0') is None

    def test_the_timing_stage_clears_stale_checkpoints(self):
        source = inspect.getsource(test_phase2_pipeline.run_stage_timing)
        assert 'shutil.rmtree' in source
        assert 'checkpoints' in source

    def test_the_first_arm_is_budget_checked_against_the_pipeline_estimate(self):
        """Without a seed the first run is waved through with any time left and killed mid-epoch,
        which loses the report the whole reserve exists to protect."""
        signature = inspect.signature(test_phase2_pipeline.run_stage_timing)
        assert 'trial_estimate' in signature.parameters
        source = inspect.getsource(test_phase2_pipeline.run_stage_timing)
        assert 'estimate = trial_estimate' in source


def test_a_passing_timing_check_carries_no_detail():
    """`Report.record` prints detail on pass as well as fail, so a success must supply none.
    Building the failure explanation unconditionally labelled good runs as having printed no
    timing line."""
    source = inspect.getsource(test_phase2_pipeline.run_stage_timing)
    tree = ast.parse(textwrap.dedent(source))
    # The detail branches must sit under `if not ok`, not run on every path.
    guarded = [node for node in ast.walk(tree.body[0])
               if isinstance(node, ast.If)
               and isinstance(node.test, ast.UnaryOp)
               and isinstance(node.test.op, ast.Not)
               and getattr(node.test.operand, 'id', None) == 'ok'
               and any('printed no' in getattr(c, 'value', '')
                       for c in ast.walk(node) if isinstance(c, ast.Constant)
                       and isinstance(c.value, str))]
    assert guarded, 'the "printed no timing line" message is not guarded by `if not ok`'
