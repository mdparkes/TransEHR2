"""Probes for the sparse text embedding path.

Text embeddings are stored sparsely and were densified per episode inside `MixedDataset` before
collation, which put roughly 860 MiB of mostly-zero tensor through the worker boundary and one
host-side copy for every batch of 200. They are now carried sparsely and densified on the device
the batch lands on.

What has to hold is that the tensor the model receives is bit-identical to the one the old path
produced, including where the crop window drops notes. These probes rebuild the old
densification directly from the stored arrays and compare.
"""

import numpy as np
import pytest
import torch

from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import densify_text_embeddings


TS_LEN, EMBED_DIM, N_TEXT = 12, 5, 2


def reference_dense(offsets, timesteps, embeddings, idx, ts_start, ts_end, embed_dim):
    """The densification `MixedDataset.__getitem__` used to do, kept as the oracle."""
    out = np.zeros((ts_end - ts_start, embed_dim), dtype=np.float32)
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    for i in range(end - start):
        ts = int(timesteps[start:end][i])
        if ts_start <= ts < ts_end:
            out[ts - ts_start] = embeddings[start:end][i]
    return out


def sparse_entry(offsets, timesteps, embeddings, idx, ts_start, ts_end, embed_dim):
    """What `__getitem__` emits now: indices in the cropped frame, plus their rows."""
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    steps = np.asarray(timesteps[start:end])
    keep = (steps >= ts_start) & (steps < ts_end)
    return (
        torch.from_numpy((steps[keep] - ts_start).astype(np.int64)),
        torch.from_numpy(np.asarray(embeddings[start:end])[keep].astype(np.float32)),
    )


def build_feature(note_counts, rng, embed_dim=EMBED_DIM, ts_len=TS_LEN):
    """One text feature's stored arrays, given how many notes each episode has."""
    offsets = np.concatenate([[0], np.cumsum(note_counts)]).astype(np.int64)
    total = int(offsets[-1])
    timesteps = np.concatenate(
        [rng.choice(ts_len, size=n, replace=False) for n in note_counts if n]
        or [np.zeros(0)]
    ).astype(np.int64)
    embeddings = rng.standard_normal((total, embed_dim)).astype(np.float32)
    return offsets, timesteps, embeddings


def make_batch(note_counts_per_feature, ts_start, ts_end, seed=0):
    """Collate a batch through the new path and densify it, plus the oracle for comparison."""
    rng = np.random.default_rng(seed)
    n_episodes = len(note_counts_per_feature[0])
    features = [build_feature(counts, rng) for counts in note_counts_per_feature]
    cropped_len = ts_end - ts_start

    episodes = []
    for idx in range(n_episodes):
        episodes.append({
            'val_numeric_indicators': torch.zeros(cropped_len, 1),
            'val_numeric_values': [torch.zeros(cropped_len, 1)],
            'val_categorical_indicators': torch.zeros(cropped_len, 1),
            'val_categorical_values': [torch.zeros(cropped_len, 1)],
            'val_ordinal_indicators': torch.zeros(cropped_len, 0),
            'val_ordinal_values': [],
            'val_multilabel_indicators': torch.zeros(cropped_len, 0),
            'val_multilabel_values': [],
            'val_text_indicators': torch.zeros(cropped_len, len(features)),
            'val_text_embeddings': [
                sparse_entry(*feature, idx, ts_start, ts_end, EMBED_DIM)
                for feature in features
            ],
            'val_times': torch.zeros(cropped_len),
            'val_masks': torch.ones(cropped_len),
            'event_indicators': torch.zeros(cropped_len, 1),
            'event_times': torch.zeros(cropped_len),
            'event_masks': torch.ones(cropped_len),
            'static_data': torch.zeros(2),
            'mortality': torch.tensor(0.0),
            'length_of_stay': torch.tensor(0.0),
            'phenotype': torch.zeros(1),
        })

    oracle = torch.from_numpy(np.stack([
        np.stack([
            reference_dense(*feature, idx, ts_start, ts_end, EMBED_DIM)
            for feature in features
        ], axis=1)
        for idx in range(n_episodes)
    ], axis=0))

    collated = collate_tensorized(episodes, history_len_steps=0)
    return collated, oracle


@pytest.mark.parametrize('note_counts', [
    pytest.param([[0, 0, 0], [0, 0, 0]], id='no episode has a note'),
    pytest.param([[1, 1, 1], [1, 1, 1]], id='one note each'),
    pytest.param([[0, 1, 4], [3, 0, 1]], id='mixed, several notes'),
    pytest.param([[5, 0, 0], [0, 0, 5]], id='all notes on one episode'),
])
def test_densified_tensor_is_bit_identical_to_the_old_path(note_counts):
    collated, oracle = make_batch(note_counts, ts_start=0, ts_end=TS_LEN)
    densified = densify_text_embeddings(collated)
    actual = densified['val_data']['text']['embedded_values']
    assert actual.shape == oracle.shape
    assert torch.equal(actual, oracle)


@pytest.mark.parametrize('window', [(0, TS_LEN), (4, TS_LEN), (0, 8), (5, 9)])
def test_notes_outside_the_crop_window_are_dropped_identically(window):
    """`HISTORY_LEN_STEPS` crops the axis, and a note outside it must vanish from both paths."""
    ts_start, ts_end = window
    collated, oracle = make_batch([[3, 2, 4], [1, 5, 0]], ts_start, ts_end, seed=7)
    actual = densify_text_embeddings(collated)['val_data']['text']['embedded_values']
    assert actual.shape[1] == ts_end - ts_start
    assert torch.equal(actual, oracle)


def test_the_sparse_keys_do_not_survive_densification():
    """Downstream code branches on which keys the text dict has, so the intermediate form must
    not leak past `move_batch_to_device`."""
    collated, _ = make_batch([[1, 2, 0], [0, 1, 1]], 0, TS_LEN)
    text = densify_text_embeddings(collated)['val_data']['text']
    assert 'embedded_values' in text
    assert 'sparse_embeddings' not in text
    assert 'sparse_dense_shape' not in text


def test_densifying_twice_is_a_no_op():
    """`move_batch_to_device` is called once per batch per loop, but a caller that densified
    earlier must not have its tensor rebuilt or destroyed."""
    collated, oracle = make_batch([[2, 0, 1], [1, 1, 1]], 0, TS_LEN)
    once = densify_text_embeddings(collated)
    tensor = once['val_data']['text']['embedded_values']
    twice = densify_text_embeddings(once)
    assert twice['val_data']['text']['embedded_values'] is tensor
    assert torch.equal(tensor, oracle)


def test_a_batch_without_text_is_untouched():
    assert densify_text_embeddings({'val_data': {'numeric': {}}}) == {
        'val_data': {'numeric': {}}}


def test_the_collated_batch_is_far_smaller_than_the_dense_form():
    """The entire point: the sparse block must not scale with the timestep axis."""
    collated, oracle = make_batch([[1, 1, 1], [1, 1, 1]], 0, TS_LEN)
    blocks = collated['val_data']['text']['sparse_embeddings']
    sparse_bytes = sum(
        t.numel() * t.element_size() for block in blocks for t in block.values())
    dense_bytes = oracle.numel() * oracle.element_size()
    assert sparse_bytes < dense_bytes / 4, f'{sparse_bytes} vs {dense_bytes}'
