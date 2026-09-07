"""Probes for restricting a run to a cohort.

Two claims are load-bearing. The row subset must not disturb what an episode contains: the
figures compare models across arms, so if narrowing the cohort also changed the tensors, every
contrast would confound the two. And the cohort predicate must not admit an episode whose only
pre-admission record is one the model never reads, since such an episode is exactly the
dilution the cohort exists to remove.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_seq_len_crop import _assert_items_equal, _build_arrays, EPISODE_SPECS, H_SMALL, E_SMALL

from TransEHR2.data.cohorts import cohort_indices, cohort_mask
from TransEHR2.data.datasets import MixedDataset


SUBSET = [1, 3, 4]


def _dataset(**overrides):
    arrays = _build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL)
    arrays.update(overrides)
    return MixedDataset(**arrays)


def test_the_subset_narrows_the_index_and_nothing_else():
    """Episode i of the subset must be byte-for-byte episode SUBSET[i] of the whole."""
    whole = _dataset()
    part = _dataset(episode_indices=np.array(SUBSET))
    assert len(whole) == len(EPISODE_SPECS)
    assert len(part) == len(SUBSET)
    for position, row in enumerate(SUBSET):
        _assert_items_equal(part[position], whole[row], f'subset position {position}')


def test_labels_follow_the_subset():
    """`positive_class_weight` counts these arrays directly rather than iterating the loader."""
    whole = _dataset()
    part = _dataset(episode_indices=np.array(SUBSET))
    assert np.array_equal(part.mortality, np.asarray(whole.mortality)[SUBSET])
    assert np.array_equal(part.phenotype, np.asarray(whole.phenotype)[SUBSET])
    assert np.array_equal(part.length_of_stay, np.asarray(whole.length_of_stay)[SUBSET])
    assert part.n_extracted_episodes == len(EPISODE_SPECS)


@pytest.mark.parametrize('indices,message', [
    ([], 'selects no episodes'),
    ([0, len(EPISODE_SPECS)], 'extracted arrays hold'),
    ([-1], 'extracted arrays hold'),
])
def test_an_unusable_subset_is_refused(indices, message):
    with pytest.raises(ValueError, match=message):
        _dataset(episode_indices=np.array(indices, dtype=np.int64))


def test_event_only_history_does_not_qualify_an_episode():
    """The collate slices history off the event stream, so such an episode has no readable
    history and would dilute the contrast exactly as an episode with none at all."""
    specs = [
        (0, 6, 0, 6),   # no history in either stream
        (0, 6, 3, 6),   # event history only -- invisible to the model
        (2, 6, 0, 6),   # value history only
    ]
    arrays = _build_arrays(specs, H_SMALL, E_SMALL)
    assert list(cohort_mask(arrays, 'any_history')) == [False, False, True]


def test_the_discharge_cohort_needs_the_text_in_the_history_region():
    arrays = _build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL)
    indicators = np.zeros_like(np.asarray(arrays['val_text_indicators']))
    # Episode 2 carries history; put a discharge summary on its last history timestep, and put
    # one on episode 0's first in-stay timestep, which must not qualify it.
    history_end = arrays['max_history_len_steps']
    indicators[2, history_end - 1, 0] = 1.0
    indicators[0, history_end, 0] = 1.0
    arrays['val_text_indicators'] = indicators
    mask = cohort_mask(arrays, 'discharge_summary')
    assert mask[2] and not mask[0]
    assert list(cohort_indices(arrays, 'discharge_summary')) == [2]


def test_no_cohort_means_every_episode():
    arrays = _build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL)
    assert cohort_mask(arrays, None) is None
    assert cohort_indices(arrays, None) is None
    with pytest.raises(ValueError, match='unknown cohort'):
        cohort_mask(arrays, 'everyone')
