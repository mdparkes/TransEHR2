"""Regression probe: the static width the model builds for must match the width on disk.

`MixedClassifier` concatenates the value embedding with `static_data` before its feedforward
layer, so the layer's input width is `d_model + static_width`. The entry points used to pass
`len(STATIC_FEATS)` as that width -- a feature *count*. The extraction writes
`sum(static_feat_dims)`, and since 75c39e8 ("Fixed categorical/ordinal feature encoding scheme,
now actually one-hot") a categorical static occupies `size` columns rather than one.

With the shipped MIMIC-IV config that is Age (numeric, 1) + Gender (categorical, 3) = 4 columns
against a declared 2, so the first forward pass dies with

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (110000x260 and 258x256)

  110000 = batch 200 x 550 timesteps,  260 = 256 + 4 (data),  258 = 256 + 2 (model)

Arrays extracted before 75c39e8 were the old width and matched, which is why this only surfaced
on a re-extraction. These probes assert the model-side and extraction-side derivations agree,
and that the count is not a valid stand-in for the width.
"""

import os
import yaml

import pytest

from TransEHR2.constants import MAX_TOKEN_LENGTH
from TransEHR2.data.preprocessing import compute_static_feat_dims


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CONFIG = os.path.join(REPO_ROOT, 'TransEHR2', 'configs', 'datasets', 'mimic4.yaml')
VARIABLE_PROPERTIES = os.path.join(REPO_ROOT, 'data', 'variable_properties.yaml')


@pytest.fixture(scope='module')
def shipped_config():
    with open(DATASET_CONFIG) as fh:
        dataset_config = yaml.safe_load(fh)
    with open(VARIABLE_PROPERTIES) as fh:
        variable_properties = yaml.safe_load(fh)
    return dataset_config, variable_properties


def test_one_hot_static_is_wider_than_the_feature_count(shipped_config):
    """The bug in one line: for the shipped config, count != width."""
    dataset_config, variable_properties = shipped_config
    static_feats = dataset_config['STATIC_FEATS']

    width = sum(compute_static_feat_dims(
        variable_properties, static_feats, MAX_TOKEN_LENGTH
    ))
    assert width != len(static_feats), (
        'this config no longer distinguishes the two derivations, so it cannot defend against '
        'the confusion; point the probe at a config with a categorical static'
    )
    assert width == 4 and len(static_feats) == 2


def test_helper_matches_the_extraction(shipped_config):
    """`_get_tensor_dimensions` must route through the same helper the model side uses."""
    from TransEHR2.data import preprocessing
    import inspect

    source = inspect.getsource(preprocessing._get_tensor_dimensions)
    assert 'compute_static_feat_dims(' in source, (
        'the extraction has stopped using the shared helper, so the two derivations can drift '
        'apart again -- which is exactly the failure this probe exists to prevent'
    )


def test_categorical_static_contributes_its_size(shipped_config):
    _, variable_properties = shipped_config
    dims = compute_static_feat_dims(
        variable_properties, ['Age', 'Gender'], MAX_TOKEN_LENGTH
    )
    assert dims == [1, 3]


def test_text_static_contributes_the_token_length():
    variable_properties = {
        'Note': {'type': 'text', 'size': 1},
        'Age': {'type': 'numeric', 'size': 1},
    }
    dims = compute_static_feat_dims(variable_properties, ['Note', 'Age'], 8192)
    assert dims == [8192, 1]


def test_entry_points_do_not_pass_the_feature_count():
    """Guard every entry point at once: none of them may size statics by count again."""
    offenders = []
    for name in ['run_experiment_accelerate.py', 'tune_hyperparameters_accelerate.py',
                 'dump_finetuned_predictions.py',
                 os.path.join('TransEHR2', 'test_tune_hyperparameters.py')]:
        path = os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                # The word appears in explanatory comments; only a real argument counts.
                if stripped.startswith('#'):
                    continue
                if 'len(STATIC_FEATS)' in stripped:
                    offenders.append('%s:%d: %s' % (name, lineno, stripped))
    assert not offenders, (
        'static dimensions must come from compute_static_feat_dims(), not a feature count:\n'
        + '\n'.join(offenders)
    )
