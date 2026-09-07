"""Probes for the Charlson logistic regression arm.

The claim the whole comparison rests on is row correspondence. The prediction CSVs carry no
stay identifier, so the reporter pairs this arm against the in-stay-only control by row
position alone. Both arms must therefore emit one row per cohort episode, in the order
`load_dataset(..., cohort=...)` produces -- and that order is what is tested here end to end,
against arrays written in the extraction's own layout, rather than reasoned about.

The features are read out of the stored `static_data` array, whose layout is not obvious: a
categorical static is allocated `size` columns but written as a single 1-based code in the
first of them, so age and sex sit at cumulative-width offsets and sex arrives as a code rather
than a label. Both are tested, because reading the wrong column silently yields a plausible
number.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.evaluation import load_predictions, prediction_csv_path
from run_charlson_logistic_regression import (build_design_matrix, coefficient_table,
                                              fit_fold, get_fold_names, load_split,
                                              sex_encoding, write_predictions)
from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.statics import decode_categorical, static_offsets
from TransEHR2.data.preprocessing import save_dataset

HIST, EPI = 4, 3
TS = HIST + EPI
TOKEN_LEN, N_EVENT_FEATS = 8, 2

# Age (numeric, 1 column) then Gender (categorical, 3 columns) -- the dataset config's order.
STATIC_FEATS = ['Age', 'Gender']
VARIABLE_PROPERTIES = {
    'Age': {'type': 'numeric', 'size': 1, 'category_map': {}},
    'Gender': {'type': 'categorical', 'size': 3, 'category_map': {0: 'Other', 1: 'F', 2: 'M'}},
}
GENDER_MAP = VARIABLE_PROPERTIES['Gender']['category_map']

# Stored codes are 1-based, offset from the lowest key of the category map.
CODE_OTHER, CODE_F, CODE_M, CODE_MISSING = 1, 2, 3, 0


def _write_partition(directory, has_diagnosis, ages, sex_codes, mortality):
    """Write one extracted partition in the layout `save_dataset` produces.

    Args:
        directory: Partition directory to create.
        has_diagnosis: Per-episode flag; True puts a pre-admission diagnosis-descriptions
            record on the last history timestep, which is what the cohort selects on.
        ages: Per-episode age in years.
        sex_codes: Per-episode stored gender code.
        mortality: Per-episode outcome.
    """
    n = len(has_diagnosis)
    masks = np.zeros((n, TS), dtype=np.float32)
    # Every episode has one observed history timestep and one in-stay timestep, so cohort
    # membership turns on the text indicator alone rather than on the mask.
    masks[:, HIST - 1] = 1.0
    masks[:, HIST] = 1.0

    text_indicators = np.zeros((n, TS, 2), dtype=np.float32)
    for i, flagged in enumerate(has_diagnosis):
        if flagged:
            text_indicators[i, HIST - 1, 1] = 1.0

    static = np.zeros((n, 4), dtype=np.float32)
    static[:, 0] = ages
    static[:, 1] = sex_codes

    dataset = MixedDataset(
        val_numeric_indicators=np.zeros((n, TS, 1), dtype=np.float32),
        val_numeric_values=[np.zeros((n, TS, 1), dtype=np.float32)],
        val_categorical_indicators=np.zeros((n, TS, 0), dtype=np.float32),
        val_categorical_values=[],
        val_ordinal_indicators=np.zeros((n, TS, 0), dtype=np.float32),
        val_ordinal_values=[],
        val_multilabel_indicators=np.zeros((n, TS, 0), dtype=np.float32),
        val_multilabel_values=[],
        val_text_indicators=text_indicators,
        val_times=np.zeros((n, TS), dtype=np.float32),
        val_masks=masks,
        val_text_offsets=[np.zeros(n + 1, dtype=np.int64) for _ in range(2)],
        val_text_values=[np.zeros((0, TOKEN_LEN), dtype=np.int64) for _ in range(2)],
        val_text_masks=[np.zeros((0, TOKEN_LEN), dtype=np.float32) for _ in range(2)],
        val_text_timesteps=[np.zeros(0, dtype=np.int32) for _ in range(2)],
        val_text_embeddings=[],
        text_embed_dim=0,
        event_indicators=np.zeros((n, TS, N_EVENT_FEATS), dtype=np.float32),
        event_times=np.zeros((n, TS), dtype=np.float32),
        event_masks=np.zeros((n, TS), dtype=np.float32),
        static_data=static,
        mortality=np.asarray(mortality, dtype=np.float32),
        length_of_stay=np.zeros(n, dtype=np.float32),
        phenotype=np.zeros((n, 2), dtype=np.float32),
        max_ts_len=TS,
        text_token_len=[TOKEN_LEN, TOKEN_LEN],
        max_history_len_steps=HIST,
    )
    save_dataset(dataset, directory)


def _write_fold(data_dir, fold, splits):
    """Write a fold whose partitions each hold the given episodes.

    Args:
        data_dir: Directory to create the fold under.
        fold: Fold directory name.
        splits: Mapping from split name to a list of per-episode
            (episode_id, has_diagnosis, age, sex_code, mortality).
    """
    for split, episodes in splits.items():
        directory = os.path.join(data_dir, fold, split)
        os.makedirs(directory, exist_ok=True)
        _write_partition(
            directory,
            [flagged for _, flagged, _, _, _ in episodes],
            [age for _, _, age, _, _ in episodes],
            [code for _, _, _, code, _ in episodes],
            [outcome for _, _, _, _, outcome in episodes],
        )
        with open(os.path.join(data_dir, fold, f'{split}_ids.pkl'), 'wb') as handle:
            pickle.dump([episode_id for episode_id, *_ in episodes], handle)


# ---------------------------------------------------------------------------
# Reading the features out of static_data
# ---------------------------------------------------------------------------

def test_the_static_offsets_follow_the_allocated_widths_not_the_feature_count():
    """Gender is one-hot-width 3, so Age + Gender is 4 columns wide and Gender begins at 1.
    Indexing by feature position would read Gender out of Age's column."""
    offsets = static_offsets(VARIABLE_PROPERTIES, STATIC_FEATS, max_token_length=0)
    assert offsets == {'Age': 0, 'Gender': 1}


def test_a_reordered_config_moves_the_offsets():
    offsets = static_offsets(VARIABLE_PROPERTIES, ['Gender', 'Age'], max_token_length=0)
    assert offsets == {'Gender': 0, 'Age': 3}


@pytest.mark.parametrize('code,label', [
    (CODE_OTHER, 'Other'),
    (CODE_F, 'F'),
    (CODE_M, 'M'),
    (CODE_MISSING, 'Missing'),
])
def test_stored_gender_codes_decode_to_their_labels(code, label):
    assert decode_categorical([code], GENDER_MAP)[0] == label


def test_an_unmapped_gender_code_reads_as_missing_rather_than_a_category():
    assert list(decode_categorical([7, 4], GENDER_MAP)) == ['Missing', 'Missing']


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------

def test_the_design_matrix_drops_the_reference_category():
    matrix, names = build_design_matrix(
        age=[60.0, 70.0], sex_labels=['F', 'M'], charlson=[2, 5], sex_indicators=['M'],
    )
    assert names == ['age', 'charlson_index', 'sex_M']
    assert np.array_equal(matrix, np.array([[60.0, 2.0, 0.0],
                                            [70.0, 5.0, 1.0]]))


def test_the_reference_is_the_most_frequent_category():
    reference, indicators = sex_encoding(['M'] * 30 + ['F'] * 20, min_count=10)
    assert reference == 'M'
    assert indicators == ['F']


def test_a_tie_on_frequency_is_broken_alphabetically():
    """The encoding has to be a function of the data, not of the order labels arrive in."""
    assert sex_encoding(['M'] * 20 + ['F'] * 20, min_count=10)[0] == 'F'
    assert sex_encoding(['F'] * 20 + ['M'] * 20, min_count=10)[0] == 'F'


def test_a_category_too_rare_to_estimate_is_pooled_into_the_reference():
    """An unpenalized fit can drive the coefficient of a two-member category to infinity and
    predict those episodes on the strength of nothing at all."""
    labels = ['F'] * 40 + ['M'] * 30 + ['Other'] * 2
    reference, indicators = sex_encoding(labels, min_count=10)
    assert reference == 'F'
    assert indicators == ['M']

    matrix, names = build_design_matrix([60.0, 61.0, 62.0], ['F', 'M', 'Other'],
                                        [1, 2, 3], indicators)
    assert names == ['age', 'charlson_index', 'sex_M']
    # The pooled category is indistinguishable from the reference, not dropped.
    assert list(matrix[:, 2]) == [0.0, 1.0, 0.0]


def test_a_category_first_seen_outside_training_falls_into_the_reference():
    """A label the encoding never saw must still yield a row: dropping the episode would put
    this arm out of step with the control."""
    matrix, _ = build_design_matrix([60.0, 70.0], ['M', 'Unseen'], [1, 2], ['M'])
    assert matrix.shape == (2, 3)
    assert list(matrix[:, 2]) == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Row correspondence with the control arm
# ---------------------------------------------------------------------------

def test_the_split_is_the_manifest_in_array_row_order(tmp_path):
    """The manifest names episodes 1 and 3; the loaded split must be exactly those two, in
    ascending array row order, with each one's own features -- which is the order the control
    arm's inference loader produces for the same manifest."""
    data_dir = str(tmp_path / 'data')
    episodes = [
        (1001, False, 40.0, CODE_F, 0.0),
        (1002, True, 55.0, CODE_M, 1.0),
        (2001, False, 60.0, CODE_M, 0.0),
        (2002, True, 75.0, CODE_F, 1.0),
    ]
    _write_fold(data_dir, 'fold1', {'train': episodes})

    charlson = pd.Series({1002: 4, 2002: 7})
    offsets = static_offsets(VARIABLE_PROPERTIES, STATIC_FEATS, 0)
    split = load_split(data_dir, 'fold1', 'train', charlson, offsets, GENDER_MAP,
                       manifest=[1002, 2002])

    assert list(split['episode_ids']) == [1002, 2002]
    assert list(split['age']) == [55.0, 75.0]
    assert list(split['sex']) == ['M', 'F']
    assert list(split['charlson']) == [4.0, 7.0]
    assert list(split['mortality']) == [1.0, 1.0]


def test_a_cohort_episode_without_an_index_is_refused(tmp_path):
    """Dropping it would shorten this arm's CSV by one row and shift every row after it out of
    step with the control, so the comparison must fail loudly instead."""
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1001, True, 40.0, CODE_F, 0.0),
        (1002, True, 55.0, CODE_M, 1.0),
    ]})
    offsets = static_offsets(VARIABLE_PROPERTIES, STATIC_FEATS, 0)

    with pytest.raises(ValueError, match='no Charlson index'):
        load_split(data_dir, 'fold1', 'train', pd.Series({1001: 3}), offsets, GENDER_MAP,
                   manifest=[1001, 1002])


def test_ids_out_of_step_with_the_arrays_are_refused(tmp_path):
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1001, True, 40.0, CODE_F, 0.0),
        (1002, True, 55.0, CODE_M, 1.0),
    ]})
    with open(os.path.join(data_dir, 'fold1', 'train_ids.pkl'), 'wb') as handle:
        pickle.dump([1001], handle)
    offsets = static_offsets(VARIABLE_PROPERTIES, STATIC_FEATS, 0)

    with pytest.raises(ValueError, match='out of step'):
        load_split(data_dir, 'fold1', 'train', pd.Series({1001: 3, 1002: 4}), offsets,
                   GENDER_MAP, manifest=[1001, 1002])


def test_an_unextracted_partition_is_absent_rather_than_an_error(tmp_path):
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [(1001, True, 40.0, CODE_F, 0.0)]})
    offsets = static_offsets(VARIABLE_PROPERTIES, STATIC_FEATS, 0)
    assert load_split(data_dir, 'fold1', 'val', pd.Series({1001: 1}), offsets,
                      GENDER_MAP, manifest=[1001]) is None


# ---------------------------------------------------------------------------
# Fitting and output
# ---------------------------------------------------------------------------

def _synthetic_split(n, seed=0):
    """A split whose mortality follows the Charlson index, so a fit has signal to find."""
    rng = np.random.default_rng(seed)
    charlson = rng.integers(0, 10, size=n).astype(float)
    age = rng.uniform(20, 90, size=n)
    sex = np.where(rng.random(n) < 0.5, 'F', 'M')
    logit = -4.0 + 0.35 * charlson + 0.02 * age
    outcome = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return {'age': age, 'sex': sex, 'charlson': charlson, 'mortality': outcome,
            'episode_ids': np.arange(n)}


def test_the_fit_predicts_every_available_split():
    splits = {'train': _synthetic_split(400, 0), 'val': _synthetic_split(120, 1),
              'test': _synthetic_split(120, 2)}
    predictions, model, names = fit_fold(splits, ['M'], None, 1.0, None, 1000)

    assert set(predictions) == {'train', 'val', 'test'}
    for split, (probabilities, targets) in predictions.items():
        assert probabilities.shape == targets.shape == (len(splits[split]['mortality']),)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    # The index was generated with a positive effect, so its coefficient must be positive.
    assert model.coef_[0][names.index('charlson_index')] > 0


def test_an_absent_split_is_skipped_rather_than_predicted():
    splits = {'train': _synthetic_split(200, 0), 'val': None, 'test': _synthetic_split(60, 2)}
    predictions, _, _ = fit_fold(splits, ['M'], None, 1.0, None, 1000)
    assert set(predictions) == {'train', 'test'}


def test_a_single_outcome_training_split_is_refused():
    """A split where nobody died leaves the intercept unbounded; the fit would run and the
    predictions would be meaningless."""
    split = _synthetic_split(100, 0)
    split['mortality'] = np.zeros_like(split['mortality'])
    with pytest.raises(ValueError, match='not identifiable'):
        fit_fold({'train': split}, ['M'], None, 1.0, None, 1000)


def test_the_predictions_are_written_where_the_reporter_reads_them(tmp_path):
    """The reporter locates a column by experiment number and reads `pred`/`target` columns, so
    the path and the header are the contract this arm has to meet."""
    model_dir = str(tmp_path / 'models')
    experiment = 'experiment19_charlson_logreg_charlsonsubset_rev'
    probabilities = np.array([0.1, 0.9, 0.4])
    targets = np.array([0.0, 1.0, 1.0])

    path = write_predictions(model_dir, experiment, 'fold1', 'test', probabilities, targets)
    assert path == prediction_csv_path(os.path.join(model_dir, experiment), 'fold1',
                                       'mortality', 'test')

    loaded_predictions, loaded_targets, _ = load_predictions(
        os.path.join(model_dir, experiment), 'fold1', 'mortality', 'test'
    )
    assert np.allclose(loaded_predictions.ravel(), probabilities)
    assert np.allclose(loaded_targets.ravel(), targets)


def test_the_coefficient_table_reports_the_intercept_and_odds_ratios():
    splits = {'train': _synthetic_split(300, 0)}
    _, model, names = fit_fold(splits, ['M'], None, 1.0, None, 1000)
    table = coefficient_table(model, names)

    assert list(table['term']) == ['intercept', *names]
    assert np.allclose(table['odds_ratio'], np.exp(table['coefficient']))


def test_fold_zero_is_reserved_for_tuning(tmp_path):
    for fold in ('fold0', 'fold1', 'fold2', 'fold10', 'notafold'):
        (tmp_path / fold).mkdir()
    assert get_fold_names(str(tmp_path)) == ['fold1', 'fold2', 'fold10']
