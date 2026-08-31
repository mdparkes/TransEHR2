"""Regression probe: a feature type the extraction produced no features for.

`load_dataset` reads the per-type feature counts from `metadata.pkl`, and for the two optional
types it does so with a default:

    n_ord = metadata.get('n_ordinal_feats', 0)
    n_ml  = metadata.get('n_multilabel_feats', 0)

so a count of zero arises either because the extraction genuinely produced none, or because the
key is absent from the metadata. Either way `load_dataset` substitutes a `(0, 0, 0)` array, which
cannot be sliced per episode.

The substitute must still be two-dimensional `(timesteps, features)`. `collate_tensorized` stacks
per-episode indicators into `(batch, timesteps, features)`, and `_gen_val_assoc_feat_mask` unpacks
exactly three dimensions from that. A bare `torch.empty(0)` collates to `(batch, 0)` and the first
pretraining batch dies with `not enough values to unpack (expected 3, got 2)` -- observed on the
re-extracted MIMIC-IV arrays, on both encoding arms, roughly eight minutes into the Phase 2 smoke
test's memory stage.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_seq_len_crop import _build_arrays, EPISODE_SPECS, H_SMALL, E_SMALL

from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import generate_record_masks


EMPTY_TYPES = ['ordinal', 'multilabel']


def _dataset_without(feature_type):
    """Build a dataset in which `feature_type` has no features, as `load_dataset` would."""
    kwargs = _build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL)
    kwargs['val_%s_indicators' % feature_type] = np.empty((0, 0, 0), dtype=np.float32)
    kwargs['val_%s_values' % feature_type] = []
    return MixedDataset(**kwargs)


@pytest.mark.parametrize('feature_type', EMPTY_TYPES)
def test_absent_feature_type_keeps_the_timestep_axis(feature_type):
    dataset = _dataset_without(feature_type)
    indicators = dataset[0]['val_%s_indicators' % feature_type]
    assert indicators.ndim == 2, (
        'a per-episode indicator tensor must stay (timesteps, features) even with no features; '
        'a 1-D tensor collates to (batch, features) and loses the timestep axis'
    )
    assert tuple(indicators.shape) == (dataset.ts_len, 0)


@pytest.mark.parametrize('feature_type', EMPTY_TYPES)
def test_absent_feature_type_collates_to_three_dimensions(feature_type):
    dataset = _dataset_without(feature_type)
    batch = collate_tensorized(
        [dataset[i] for i in range(len(EPISODE_SPECS))], True, H_SMALL
    )
    indicators = batch['val_data'][feature_type]['indicators']
    assert tuple(indicators.shape) == (len(EPISODE_SPECS), dataset.ts_len, 0)


@pytest.mark.parametrize('feature_type', EMPTY_TYPES)
def test_absent_feature_type_survives_record_masking(feature_type):
    """The failure the smoke test hit: masking unpacks three dimensions from the indicators."""
    dataset = _dataset_without(feature_type)
    batch = collate_tensorized(
        [dataset[i] for i in range(len(EPISODE_SPECS))], True, H_SMALL
    )
    val_masks, event_masks = generate_record_masks(batch, 0.15, 10.0, 1.0)

    assert tuple(val_masks[feature_type]['indicators'].shape) == (
        len(EPISODE_SPECS), dataset.ts_len, 0
    )
    assert val_masks[feature_type]['values'] == []
    # The populated types are untouched by a neighbouring type being empty.
    assert val_masks['numeric']['indicators'].shape[-1] > 0
    assert event_masks is not None
