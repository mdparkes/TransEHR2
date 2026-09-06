"""Probes for the length-sorted batching in embed_text.py.

The batcher permutes its input so each batch can be trimmed to its own longest sequence. That
is worth doing -- attention is quadratic in sequence length and every note is stored padded to
the full 8192-token context -- but its failure mode is silent. A permutation that loses track
of where a row belongs writes embeddings to the wrong episodes, and nothing downstream errors:
the arrays have the right shape and the model simply learns from mismatched text.

So the property to hold is not that the batcher is fast, but that scattering its output by the
indices it yields reconstructs exactly what embedding the rows in storage order would have
produced.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed_text import length_sorted_batches


def _corpus(n=37, width=64, seed=0):
    """Padded token rows of assorted real lengths, as the extraction stores them."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, width + 1, size=n)
    tokens = np.zeros((n, width), dtype=np.int64)
    masks = np.zeros((n, width), dtype=np.int64)
    for row, length in enumerate(lengths):
        tokens[row, :length] = rng.integers(5, 1000, size=length)
        masks[row, :length] = 1
    return tokens, masks, lengths


@pytest.mark.parametrize('batch_size', [1, 4, 8, 64])
def test_every_row_is_emitted_exactly_once(batch_size):
    tokens, masks, _ = _corpus()
    indices = np.arange(len(tokens)) * 3 + 11  # destinations are not 0..n-1
    seen = [i for batch, _, _ in length_sorted_batches(tokens, masks, indices, batch_size)
            for i in batch]
    assert sorted(seen) == sorted(indices.tolist())


@pytest.mark.parametrize('batch_size', [1, 4, 8, 64])
def test_a_row_travels_with_its_own_destination(batch_size):
    """The scatter is by yielded index, so a row and its index must not come apart."""
    tokens, masks, lengths = _corpus()
    indices = np.arange(len(tokens)) * 3 + 11
    destination = {int(index): row for row, index in enumerate(indices)}
    for batch, batch_tokens, batch_masks in length_sorted_batches(
        tokens, masks, indices, batch_size
    ):
        for position, index in enumerate(batch):
            row = destination[int(index)]
            length = int(lengths[row])
            assert np.array_equal(batch_tokens[position, :length], tokens[row, :length])
            assert int(batch_masks[position].sum()) == length


@pytest.mark.parametrize('batch_size', [4, 8])
def test_the_trim_removes_only_padding(batch_size):
    tokens, masks, _ = _corpus()
    indices = np.arange(len(tokens))
    for _, batch_tokens, batch_masks in length_sorted_batches(
        tokens, masks, indices, batch_size
    ):
        # Nothing real is cut: every row's mask still sums to its full real length.
        assert batch_tokens.shape[1] == batch_masks.shape[1]
        assert batch_tokens.shape[1] >= int(batch_masks.sum(axis=1).max())
        # Trailing columns beyond the batch maximum are gone.
        assert batch_tokens.shape[1] == max(int(batch_masks.sum(axis=1).max()), 1)


def test_sorting_makes_batches_narrower_than_storage_order():
    """The whole point: a batch in storage order is as wide as the longest note anywhere."""
    tokens, masks, _ = _corpus(n=64, width=128)
    indices = np.arange(len(tokens))
    widths = [t.shape[1] for _, t, _ in length_sorted_batches(tokens, masks, indices, 8)]
    assert max(widths) <= tokens.shape[1]
    assert sum(widths) < len(widths) * tokens.shape[1]


def test_an_all_padding_batch_keeps_a_column():
    tokens = np.zeros((3, 16), dtype=np.int64)
    masks = np.zeros((3, 16), dtype=np.int64)
    batches = list(length_sorted_batches(tokens, masks, np.arange(3), 3))
    assert batches[0][1].shape[1] == 1


def test_episode_ids_are_found_beside_the_partition(tmp_path):
    """extract_mimic writes them at {fold}/{partition}_ids.pkl, not inside the partition.

    Looking only inside the partition returned None for every partition, so the cache key fell
    back to the partition path -- unique by construction -- and no text was ever reused across
    folds. The job did six times the embedding work with nothing in its output to say so.
    """
    import pickle
    from embed_text import load_episode_ids

    fold = tmp_path / 'fold1'
    (fold / 'train').mkdir(parents=True)
    ids = [1001, 1002, 1003]
    with open(fold / 'train_ids.pkl', 'wb') as handle:
        pickle.dump(ids, handle)

    assert load_episode_ids(str(fold / 'train')) == ids


def test_missing_episode_ids_are_reported_as_absent(tmp_path):
    from embed_text import load_episode_ids

    fold = tmp_path / 'fold2'
    (fold / 'test').mkdir(parents=True)
    assert load_episode_ids(str(fold / 'test')) is None
