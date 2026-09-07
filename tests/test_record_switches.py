"""Probes for the three switches that decide which records reach the model.

The experiments they serve differ only in these flags, so a flag that does not do exactly what
its name says silently turns one experiment into another. Two properties carry the weight.

With every switch on, the batch must be byte-for-byte what it was before the switches existed:
the tuned configuration and every finished phase run through this path, and a change there
would invalidate results that are already recorded.

When a switch is off, the records it names must leave by every route. Text records reach the
encoder through a sparse embedding block that is separate from the indicator tensor, so
clearing the indicator alone would drop a record from the mask while still feeding its
embedding to the model.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_seq_len_crop import _build_arrays, EPISODE_SPECS, H_SMALL, E_SMALL

from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.preprocessing import collate_tensorized


def _collate(**switches):
    dataset = MixedDataset(**_build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL))
    batch = [dataset[i] for i in range(len(EPISODE_SPECS))]
    return collate_tensorized(batch, history_len_steps=dataset.history_len_steps, **switches)


def _nontext_indicators(collated):
    return [collated['val_data'][kind]['indicators']
            for kind in ('numeric', 'categorical', 'ordinal', 'multilabel')]


def test_every_switch_on_changes_nothing():
    """The tuned configuration runs this path; it must be the pre-existing default."""
    default = _collate()
    explicit = _collate(use_historical_nontext_records=True, use_historical_text_records=True,
                        use_instay_records=True)
    assert torch.equal(default['val_data']['masks'], explicit['val_data']['masks'])
    assert torch.equal(default['event_data']['masks'], explicit['event_data']['masks'])
    for a, b in zip(_nontext_indicators(default), _nontext_indicators(explicit)):
        assert torch.equal(a, b)
    assert torch.equal(default['val_data']['text']['indicators'],
                       explicit['val_data']['text']['indicators'])


def test_dropping_all_history_clears_the_history_region_only():
    dropped = _collate(use_historical_nontext_records=False, use_historical_text_records=False)
    default = _collate()
    masks = dropped['val_data']['masks']
    assert masks[:, :H_SMALL].sum() == 0
    assert torch.equal(masks[:, H_SMALL:], default['val_data']['masks'][:, H_SMALL:])


def test_dropping_in_stay_empties_the_event_stream_too():
    """The event stream holds in-stay records only, so it has nothing left to carry."""
    dropped = _collate(use_instay_records=False)
    assert dropped['val_data']['masks'][:, H_SMALL:].sum() == 0
    assert dropped['event_data']['masks'].sum() == 0
    assert dropped['event_data']['indicators'].sum() == 0
    assert dropped['val_data']['masks'][:, :H_SMALL].sum() > 0


def test_historical_text_only_keeps_the_text_and_nothing_else():
    """Experiment 14: in-stay records plus historical text, no historical measurements."""
    dropped = _collate(use_historical_nontext_records=False)
    for indicators in _nontext_indicators(dropped):
        if indicators.shape[-1] > 0:
            assert indicators[:, :H_SMALL].sum() == 0, 'history measurements survived'
    text = dropped['val_data']['text']['indicators'][:, :H_SMALL]
    masks = dropped['val_data']['masks'][:, :H_SMALL]
    # A history timestep survives exactly where a text record does.
    assert torch.equal(masks.bool(), text.bool().any(dim=-1))
    assert text.sum() > 0, 'the fixture has no historical text to keep'


def test_dropped_text_records_lose_their_embeddings_as_well():
    """Clearing the indicator alone would still feed the record through the sparse block."""
    default = _collate()
    assert default['val_data']['text']['sparse_embeddings'][0]['values'].numel() > 0
    dropped = _collate(use_historical_text_records=False)
    for feature in dropped['val_data']['text']['sparse_embeddings']:
        assert (feature['timestep_index'] < H_SMALL).sum() == 0
        assert feature['episode_index'].shape[0] == feature['values'].shape[0]


def test_in_stay_text_records_go_with_the_in_stay_switch():
    dropped = _collate(use_instay_records=False)
    for feature in dropped['val_data']['text']['sparse_embeddings']:
        assert (feature['timestep_index'] >= H_SMALL).sum() == 0
