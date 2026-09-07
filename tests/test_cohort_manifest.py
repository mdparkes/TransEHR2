"""Probes for restricting a run to an explicit list of patient-episode IDs.

A manifest exists so that a cohort can depend on something the extracted arrays do not carry,
and so that two arms of a comparison are put on the same episodes by construction rather than
by an argument that two predicates agree. Both halves of that are tested: the selection is by
ID and reaches episodes no array predicate would admit, and the rows come back in the array's
own order so the reporter can pair the arms by row position.

The failure this guards against is silent. A manifest built against a different extraction, or
an IDs file out of step with the arrays, would still produce a dataset of plausible size -- so
the mismatches are made into errors rather than left to be noticed downstream.
"""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_charlson_logreg import CODE_F, _write_fold

from TransEHR2.data.cohorts import (cohort_indices, load_episode_manifest, manifest_mask)
from TransEHR2.data.preprocessing import load_dataset, load_episode_ids


# ---------------------------------------------------------------------------
# Reading a manifest
# ---------------------------------------------------------------------------

def test_a_manifest_file_is_read_ignoring_comments_and_blanks(tmp_path):
    path = tmp_path / 'cohort.txt'
    path.write_text('# a comment\n\n1002\n2002  \n  3001\n1002\n\n# trailing note\n')
    assert list(load_episode_manifest(str(path))) == [1002, 2002, 3001]


def test_an_inline_comment_is_stripped(tmp_path):
    path = tmp_path / 'cohort.txt'
    path.write_text('1002  # kept\n2002\n')
    assert list(load_episode_manifest(str(path))) == [1002, 2002]


def test_an_iterable_of_ids_is_accepted_directly():
    assert list(load_episode_manifest([2002, 1002, 1002])) == [1002, 2002]


def test_an_empty_manifest_is_refused(tmp_path):
    path = tmp_path / 'cohort.txt'
    path.write_text('# nothing but a comment\n\n')
    with pytest.raises(ValueError, match='names no episodes'):
        load_episode_manifest(str(path))
    with pytest.raises(ValueError, match='names no episodes'):
        load_episode_manifest([])


def test_a_manifest_that_is_not_ids_names_what_it_should_be(tmp_path):
    path = tmp_path / 'cohort.txt'
    path.write_text('1002\nepisode3\n')
    with pytest.raises(ValueError, match='patient-episode IDs'):
        load_episode_manifest(str(path))


def test_a_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episode_manifest(str(tmp_path / 'absent.txt'))


# ---------------------------------------------------------------------------
# Selecting rows
# ---------------------------------------------------------------------------

def test_rows_are_selected_by_id_in_array_order():
    """The manifest's own order is irrelevant: rows come back ascending, which is the order
    both arms' loaders iterate."""
    ids = [5001, 1002, 3003, 2004]
    assert list(manifest_mask(ids, [3003, 5001])) == [True, False, True, False]
    assert list(manifest_mask(ids, [5001, 3003])) == [True, False, True, False]


def test_an_id_the_partition_does_not_hold_is_simply_absent():
    """Folds partition the episodes, so most of a manifest is missing from any one partition.
    That is normal and must not be an error."""
    assert list(manifest_mask([1002, 2004], [1002, 9999])) == [True, False]


def test_a_manifest_matching_nothing_is_refused():
    """Silently yielding an empty dataset would surface as an unrelated failure much later."""
    with pytest.raises(ValueError, match='names none of'):
        manifest_mask([1002, 2004], [7777, 8888])


def test_a_manifest_needs_the_episode_ids_alongside_it():
    with pytest.raises(ValueError, match='episode_ids must be given'):
        cohort_indices({'val_masks': np.zeros((2, 4)), 'val_text_indicators': np.zeros((2, 4, 2)),
                        'max_history_len_steps': 2}, None, manifest=[1002])


def test_a_named_cohort_and_a_manifest_intersect():
    """Given both, an episode has to satisfy both -- the manifest cannot readmit an episode the
    predicate excludes."""
    history_len = 2
    arrays = {
        'val_masks': np.zeros((3, 6), dtype=np.float32),
        'val_text_indicators': np.zeros((3, 6, 2), dtype=np.float32),
        'max_history_len_steps': history_len,
    }
    arrays['val_masks'][:, history_len - 1] = 1.0
    # Episodes 0 and 1 carry a diagnosis record; the manifest names 1 and 2.
    arrays['val_text_indicators'][0, history_len - 1, 1] = 1.0
    arrays['val_text_indicators'][1, history_len - 1, 1] = 1.0
    ids = [1001, 1002, 1003]

    assert list(cohort_indices(arrays, 'diagnosis_history', ids, None)) == [0, 1]
    assert list(cohort_indices(arrays, None, ids, [1002, 1003])) == [1, 2]
    assert list(cohort_indices(arrays, 'diagnosis_history', ids, [1002, 1003])) == [1]


def test_no_restriction_at_all_keeps_every_episode():
    arrays = {'val_masks': np.zeros((2, 4)), 'val_text_indicators': np.zeros((2, 4, 2)),
              'max_history_len_steps': 2}
    assert cohort_indices(arrays, None, None, None) is None


# ---------------------------------------------------------------------------
# Episode ids beside the arrays
# ---------------------------------------------------------------------------

def test_the_ids_are_found_beside_the_partition(tmp_path):
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1001, True, 40.0, CODE_F, 0.0),
        (1002, False, 55.0, CODE_F, 1.0),
    ]})
    ids = load_episode_ids(os.path.join(data_dir, 'fold1', 'train'), n_episodes=2)
    assert list(ids) == [1001, 1002]


def test_ids_inside_the_partition_are_also_found(tmp_path):
    """Some older extractions wrote the file there instead."""
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [(1001, True, 40.0, CODE_F, 0.0)]})
    beside = os.path.join(data_dir, 'fold1', 'train_ids.pkl')
    inside = os.path.join(data_dir, 'fold1', 'train', 'train_ids.pkl')
    os.rename(beside, inside)
    assert list(load_episode_ids(os.path.join(data_dir, 'fold1', 'train'))) == [1001]


def test_ids_out_of_step_with_the_arrays_are_refused(tmp_path):
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1001, True, 40.0, CODE_F, 0.0),
        (1002, False, 55.0, CODE_F, 1.0),
    ]})
    with open(os.path.join(data_dir, 'fold1', 'train_ids.pkl'), 'wb') as handle:
        pickle.dump([1001], handle)
    with pytest.raises(ValueError, match='out of step'):
        load_episode_ids(os.path.join(data_dir, 'fold1', 'train'), n_episodes=2)


def test_absent_ids_are_an_error_rather_than_a_guess(tmp_path):
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [(1001, True, 40.0, CODE_F, 0.0)]})
    os.remove(os.path.join(data_dir, 'fold1', 'train_ids.pkl'))
    with pytest.raises(FileNotFoundError, match='no episode ids'):
        load_episode_ids(os.path.join(data_dir, 'fold1', 'train'))


# ---------------------------------------------------------------------------
# Through the loader, which is where both arms meet
# ---------------------------------------------------------------------------

def _fold_with_mixed_records(tmp_path):
    """A fold where the manifest and the array predicate disagree on purpose.

    Episode 1002 carries a pre-admission diagnosis record; 2002 does not, which is the case
    that motivated the manifest -- its codes can be perfectly available in `diagnoses.csv`
    while the extraction holds no record of them. 3001 carries one but is not in the cohort.
    """
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1002, True, 55.0, CODE_F, 1.0),
        (2002, False, 75.0, CODE_F, 1.0),
        (3001, True, 40.0, CODE_F, 0.0),
    ]})
    return data_dir


def test_the_loader_selects_the_manifest_and_reaches_records_no_predicate_would(tmp_path):
    data_dir = _fold_with_mixed_records(tmp_path)
    base = os.path.join(data_dir, 'fold1', 'train')

    dataset = load_dataset(base, cohort_episodes=[1002, 2002])
    assert list(dataset.episode_indices) == [0, 1]
    assert len(dataset) == 2
    # 2002 has no diagnosis record, so the array predicate would have dropped it.
    predicate = load_dataset(base, cohort='diagnosis_history')
    assert list(predicate.episode_indices) == [0, 2]


def test_both_arms_given_one_manifest_select_the_same_rows(tmp_path):
    """The guarantee the manifest exists for. Whatever each arm reads -- one may drop history
    entirely, as the in-stay-only control does -- the row subset and its order are identical,
    which is what lets the reporter pair predictions by position."""
    data_dir = _fold_with_mixed_records(tmp_path)
    base = os.path.join(data_dir, 'fold1', 'train')
    manifest = [1002, 2002]

    control = load_dataset(base, cohort_episodes=manifest, history_len_steps=0)
    charlson_arm = load_dataset(base, cohort_episodes=manifest)

    assert list(control.episode_indices) == list(charlson_arm.episode_indices)
    ids = load_episode_ids(base, n_episodes=control.n_extracted_episodes)
    assert list(ids[control.episode_indices]) == manifest
    # The labels follow the same subset, so a target column cannot drift between the arms.
    assert list(control.mortality) == list(charlson_arm.mortality)


def test_the_loader_refuses_a_manifest_from_another_extraction(tmp_path):
    data_dir = _fold_with_mixed_records(tmp_path)
    with pytest.raises(ValueError, match='names none of'):
        load_dataset(os.path.join(data_dir, 'fold1', 'train'), cohort_episodes=[999001])


def test_a_manifest_path_works_through_the_loader(tmp_path):
    """The config names a path, not a list, so that is the form the loader has to accept."""
    data_dir = _fold_with_mixed_records(tmp_path)
    path = tmp_path / 'cohort.txt'
    path.write_text('# Charlson cohort\n1002\n2002\n')
    dataset = load_dataset(os.path.join(data_dir, 'fold1', 'train'), cohort_episodes=str(path))
    assert list(dataset.episode_indices) == [0, 1]
