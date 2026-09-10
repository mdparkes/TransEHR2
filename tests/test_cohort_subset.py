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


def _text_arrays(history_len, n_episodes, n_text_feats):
    """Minimal arrays for the text cohort predicates.

    `cohort_mask` reads only the value-stream mask, the text presence indicators and the width
    of the history region, and accepts them as a dict, so a full episode fixture is not needed
    to probe which record qualifies an episode.
    """
    ts_len = history_len + 4
    return {
        'val_masks': np.zeros((n_episodes, ts_len), dtype=np.float32),
        'val_text_indicators': np.zeros((n_episodes, ts_len, n_text_feats), dtype=np.float32),
        'max_history_len_steps': history_len,
    }


def test_the_diagnosis_cohort_keys_off_its_own_text_feature():
    """The two text features are separate cohorts: a discharge summary does not qualify an
    episode for the index, which is computed from coded diagnoses, and vice versa."""
    history_len = 3
    arrays = _text_arrays(history_len, 4, 2)
    arrays['val_masks'][:, history_len - 1] = 1.0

    # Episode 0: summary only. 1: diagnoses only. 2: both. 3: neither.
    arrays['val_text_indicators'][0, history_len - 1, 0] = 1.0
    arrays['val_text_indicators'][1, history_len - 1, 1] = 1.0
    arrays['val_text_indicators'][2, history_len - 1, :] = 1.0

    assert list(cohort_mask(arrays, 'diagnosis_history')) == [False, True, True, False]
    assert list(cohort_mask(arrays, 'discharge_summary')) == [True, False, True, False]
    assert list(cohort_indices(arrays, 'diagnosis_history')) == [1, 2]


def test_an_in_stay_diagnosis_record_does_not_qualify_an_episode():
    """The index scores an *earlier* admission. A record at or after the admission timestep is
    the current stay's own coding, which is what the extraction blanks."""
    history_len = 3
    arrays = _text_arrays(history_len, 2, 2)
    arrays['val_masks'][:, history_len] = 1.0
    arrays['val_text_indicators'][0, history_len, 1] = 1.0      # first in-stay timestep
    arrays['val_masks'][1, history_len - 1] = 1.0
    arrays['val_text_indicators'][1, history_len - 1, 1] = 1.0  # last history timestep

    assert list(cohort_mask(arrays, 'diagnosis_history')) == [False, True]


def test_an_unobserved_timestep_does_not_qualify_an_episode():
    """Presence is the intersection of the indicator with the observed mask, so an indicator
    left set on a padding timestep must not admit the episode."""
    history_len = 3
    arrays = _text_arrays(history_len, 1, 2)
    arrays['val_text_indicators'][0, history_len - 1, 1] = 1.0  # indicator, but mask is zero

    assert list(cohort_mask(arrays, 'diagnosis_history')) == [False]


def test_the_diagnosis_cohort_needs_the_feature_to_exist():
    """An extraction with one text feature cannot serve the index; the error names the config
    key to look at rather than reporting an index error."""
    arrays = _text_arrays(3, 2, 1)
    with pytest.raises(ValueError, match='TEXT_FEATS'):
        cohort_mask(arrays, 'diagnosis_history')


# ------------------------------------------------------------------------------------------
# The text cohort the experiments run on
# ------------------------------------------------------------------------------------------

def test_the_text_cohort_is_the_union_of_the_text_features():
    """`any_text` replaces the two single-feature cohorts, which split a population too small
    to divide and asked the same question twice."""
    history_len = 3
    arrays = _text_arrays(history_len, 4, 2)
    arrays['val_masks'][:, history_len - 1] = 1.0

    # Episode 0: summary only. 1: diagnoses only. 2: both. 3: neither.
    arrays['val_text_indicators'][0, history_len - 1, 0] = 1.0
    arrays['val_text_indicators'][1, history_len - 1, 1] = 1.0
    arrays['val_text_indicators'][2, history_len - 1, :] = 1.0

    assert list(cohort_mask(arrays, 'any_text')) == [True, True, True, False]
    assert list(cohort_indices(arrays, 'any_text')) == [0, 1, 2]


def test_the_text_cohort_needs_the_text_before_the_cutoff():
    """The history region is the pre-admission era, so a record in the episode region is
    peri-stay text and no arm reads it."""
    history_len = 3
    arrays = _text_arrays(history_len, 2, 2)
    arrays['val_masks'][:, history_len] = 1.0                   # first episode-region timestep
    arrays['val_text_indicators'][0, history_len, 0] = 1.0

    assert list(cohort_mask(arrays, 'any_text')) == [False, False]


def test_the_text_cohort_needs_an_observed_timestep():
    """An indicator on a padding timestep is not a record."""
    history_len = 3
    arrays = _text_arrays(history_len, 2, 2)
    arrays['val_text_indicators'][0, history_len - 1, 0] = 1.0  # mask left at zero

    assert list(cohort_mask(arrays, 'any_text')) == [False, False]


def test_the_text_cohort_is_contained_in_the_history_cohort():
    """Text before the cutoff is itself a pre-admission record, so the containment holds by
    construction and the Euler diagram stays nested."""
    history_len = 3
    arrays = _text_arrays(history_len, 4, 2)
    arrays['val_masks'][:, history_len - 1] = 1.0
    arrays['val_text_indicators'][0, history_len - 1, 0] = 1.0
    arrays['val_text_indicators'][1, history_len - 1, 1] = 1.0

    text = cohort_mask(arrays, 'any_text')
    history = cohort_mask(arrays, 'any_history')
    assert not (text & ~history).any(), 'an episode is in the text cohort but not in history'


def test_the_text_cohort_tolerates_an_extraction_with_no_text():
    """A no-text extraction has an empty cohort rather than an indexing error."""
    history_len = 3
    arrays = _text_arrays(history_len, 3, 0)
    arrays['val_masks'][:, history_len - 1] = 1.0

    assert list(cohort_mask(arrays, 'any_text')) == [False, False, False]
