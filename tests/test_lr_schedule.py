"""Probes for the learning rate schedule and the early stopping threshold."""

import math
import os
import sys

import pytest
import torch.utils.tensorboard  # noqa: F401  -- routines_accelerate reads it at import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TransEHR2.routines_accelerate import (
    IMPROVEMENT_THRESHOLD,
    is_improvement,
    resolve_decay_factor,
)


def test_no_half_life_leaves_the_rate_constant():
    assert resolve_decay_factor(None) == 1.0
    assert resolve_decay_factor(0) == 1.0
    assert resolve_decay_factor(-10) == 1.0


@pytest.mark.parametrize('half_life', [1, 10, 40, 132, 329])
def test_the_rate_halves_after_one_half_life(half_life):
    gamma = resolve_decay_factor(half_life)
    assert gamma ** half_life == pytest.approx(0.5)


def test_a_longer_half_life_decays_more_slowly():
    assert resolve_decay_factor(40) < resolve_decay_factor(100) < 1.0


def test_the_schedule_matches_the_continuous_form():
    # lr(e) = lr0 * 0.5 ** (e / H), which is what makes the half-life the whole schedule.
    half_life, epochs = 40.0, 97
    assert resolve_decay_factor(half_life) ** epochs == pytest.approx(0.5 ** (epochs / half_life))


def test_the_first_finite_loss_is_an_improvement():
    assert is_improvement(3.0, math.inf)


def test_an_equal_loss_is_not_an_improvement():
    assert not is_improvement(3.0, 3.0)


def test_an_improvement_must_clear_the_threshold():
    best = 2.0
    assert not is_improvement(best * (1 - IMPROVEMENT_THRESHOLD / 2), best)
    assert is_improvement(best * (1 - IMPROVEMENT_THRESHOLD * 2), best)


def test_the_threshold_is_relative_to_the_incumbent():
    # The same absolute step counts at a small loss and not at a large one.
    step = 1e-3
    assert is_improvement(1.0 - step, 1.0)
    assert not is_improvement(1000.0 - step, 1000.0)


def test_a_negative_loss_still_has_to_get_smaller():
    # best * (1 - threshold) inverts below zero and would accept a worse value.
    assert not is_improvement(-9.9999, -10.0)
    assert is_improvement(-10.1, -10.0)


def test_a_non_finite_loss_never_wins():
    assert not is_improvement(math.nan, 3.0)
    assert not is_improvement(math.inf, 3.0)
    assert not is_improvement(math.nan, math.inf)
