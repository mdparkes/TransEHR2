"""
Tests for runtime sequence-length cropping (HISTORY_LEN_STEPS / EPISODE_LEN_STEPS).

The central claim these tests defend is the equivalence that makes the sequence-length sweep
possible without re-extraction:

    cropping a dataset extracted at (H_big, E_big) down to (H_small, E_small)
    == extracting that same data at (H_small, E_small)

That holds because `filter_timeseries_records()` keeps the *most recent* historic records and
the *earliest* in-stay records, and `extract_mimic()` writes history right-justified in
[0, H) with in-stay data left-justified from index H. So both caps reduce to one contiguous
slice of the stored timestep axis.

Rather than trusting that argument, the tests below synthesise episodes in the documented
extraction layout at two different sizes and assert that the cropped large dataset is
element-for-element identical to the natively-small one -- including the sparse text storage,
whose timestep indices have to be filtered and re-based rather than sliced.
"""

import numpy as np
import os
import pickle
import shutil
import tempfile
import torch

from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.preprocessing import (
    collate_tensorized,
    get_text_counts_from_dataset_vectorized,
    load_dataset,
    prepare_dataloaders,
    save_dataset,
)


# Extraction-time capacity used for the "large" dataset, and the smaller limits to crop to.
H_BIG, E_BIG = 8, 6
H_SMALL, E_SMALL = 3, 4

# (n_val_history, n_val_episode, n_event_history, n_event_episode) per episode.
# Value- and event-associated data are filtered independently at extraction time, so their
# history lengths differ per episode; the crop must stay correct for both at once.
EPISODE_SPECS = [
    (0, 6, 0, 6),     # no history at all
    (2, 6, 5, 3),     # val history below both caps, event history above the small cap
    (5, 5, 1, 6),     # val history between the two caps
    (12, 6, 9, 6),    # both histories exceed even the large cap (truncated at extraction)
    (7, 2, 7, 1),     # short in-stay stay, below both episode caps
]

NUMERIC_DIMS = [1, 2]
CATEGORICAL_DIMS = [3]
ORDINAL_DIMS = [4]
MULTILABEL_DIMS = [2]
N_EVENT_FEATS = 3
STATIC_DIM = 2
PHENOTYPE_DIM = 3
TEXT_TOKEN_LEN = 5
EMBED_DIM = 4


def _record_id(episode_idx: int, ordinal: int, n_records: int, is_history: bool) -> int:
    """Return a value unique to one record of one episode.

    History records are numbered backwards from the admission so that the *most recent* ones
    (the ones truncation keeps) carry stable ids regardless of how many are retained.
    """
    base = (episode_idx + 1) * 1000
    if is_history:
        return base - (n_records - ordinal)
    return base + 100 + ordinal


def _has_text(record_id: int) -> bool:
    return record_id % 3 == 0


def _build_arrays(specs, max_history_len_steps, max_episode_len_steps):
    """Synthesise extracted arrays in the layout `extract_mimic()` produces.

    Args:
        specs: Per-episode (n_val_history, n_val_episode, n_event_history, n_event_episode).
        max_history_len_steps: History region width, i.e. MAX_HISTORY_LEN_STEPS.
        max_episode_len_steps: In-stay region width, i.e. MAX_EPISODE_LEN_STEPS.

    Returns:
        Dict of keyword arguments for `MixedDataset`.
    """
    n = len(specs)
    hist_len = max_history_len_steps
    ts_len = max_history_len_steps + max_episode_len_steps

    val_times = np.zeros((n, ts_len), dtype=np.float32)
    val_masks = np.zeros((n, ts_len), dtype=np.float32)
    numeric_ind = np.zeros((n, ts_len, len(NUMERIC_DIMS)), dtype=np.float32)
    numeric_vals = [np.zeros((n, ts_len, d), dtype=np.float32) for d in NUMERIC_DIMS]
    categorical_ind = np.zeros((n, ts_len, len(CATEGORICAL_DIMS)), dtype=np.float32)
    categorical_vals = [np.zeros((n, ts_len, d), dtype=np.int64) for d in CATEGORICAL_DIMS]
    ordinal_ind = np.zeros((n, ts_len, len(ORDINAL_DIMS)), dtype=np.float32)
    ordinal_vals = [np.zeros((n, ts_len, d), dtype=np.int64) for d in ORDINAL_DIMS]
    multilabel_ind = np.zeros((n, ts_len, len(MULTILABEL_DIMS)), dtype=np.float32)
    multilabel_vals = [np.zeros((n, ts_len, d), dtype=np.float32) for d in MULTILABEL_DIMS]
    text_ind = np.zeros((n, ts_len, 1), dtype=np.float32)
    event_times = np.zeros((n, ts_len), dtype=np.float32)
    event_masks = np.zeros((n, ts_len), dtype=np.float32)
    event_ind = np.zeros((n, ts_len, N_EVENT_FEATS), dtype=np.float32)
    static = np.zeros((n, STATIC_DIM), dtype=np.float32)
    mortality = np.zeros(n, dtype=np.float32)
    length_of_stay = np.zeros(n, dtype=np.float32)
    phenotype = np.zeros((n, PHENOTYPE_DIM), dtype=np.float32)

    text_timesteps, text_values, text_masks_sparse, text_embeddings, text_counts = [], [], [], [], []

    def fill_val(i, idx, rid):
        val_times[i, idx] = rid * 0.5
        val_masks[i, idx] = 1.0
        for f, d in enumerate(NUMERIC_DIMS):
            numeric_ind[i, idx, f] = float((rid + f) % 2)
            numeric_vals[f][i, idx, :] = [rid + 0.1 * f + 0.01 * k for k in range(d)]
        for f, d in enumerate(CATEGORICAL_DIMS):
            categorical_ind[i, idx, f] = 1.0
            categorical_vals[f][i, idx, :] = [(rid + k) % 7 for k in range(d)]
        for f, d in enumerate(ORDINAL_DIMS):
            ordinal_ind[i, idx, f] = 1.0
            ordinal_vals[f][i, idx, :] = [(rid + k) % 5 for k in range(d)]
        for f, d in enumerate(MULTILABEL_DIMS):
            multilabel_ind[i, idx, f] = 1.0
            multilabel_vals[f][i, idx, :] = [float((rid + k) % 3) for k in range(d)]
        if _has_text(rid):
            text_ind[i, idx, 0] = 1.0

    def fill_event(i, idx, rid):
        event_times[i, idx] = rid * 0.25
        event_masks[i, idx] = 1.0
        event_ind[i, idx, :] = [float((rid + k) % 2) for k in range(N_EVENT_FEATS)]

    for i, (n_vh, n_ve, n_eh, n_ee) in enumerate(specs):
        # Truncation keeps the most recent history and the earliest in-stay records.
        vh = min(n_vh, max_history_len_steps)
        ve = min(n_ve, max_episode_len_steps)
        eh = min(n_eh, max_history_len_steps)
        ee = min(n_ee, max_episode_len_steps)

        episode_text = []

        # History is right-justified within [0, hist_len).
        for m in range(vh):
            rid = _record_id(i, n_vh - vh + m, n_vh, is_history=True)
            idx = hist_len - vh + m
            fill_val(i, idx, rid)
            if _has_text(rid):
                episode_text.append((idx, rid))
        # In-stay data is left-justified from hist_len.
        for k in range(ve):
            rid = _record_id(i, k, n_ve, is_history=False)
            idx = hist_len + k
            fill_val(i, idx, rid)
            if _has_text(rid):
                episode_text.append((idx, rid))

        for m in range(eh):
            fill_event(i, hist_len - eh + m,
                       _record_id(i, n_eh - eh + m, n_eh, is_history=True))
        for k in range(ee):
            fill_event(i, hist_len + k, _record_id(i, k, n_ee, is_history=False))

        static[i, :] = [i + 0.5, i + 1.5]
        mortality[i] = float(i % 2)
        length_of_stay[i] = 10.0 + i
        phenotype[i, :] = [float((i + k) % 2) for k in range(PHENOTYPE_DIM)]

        # Sparse text entries are appended in ascending timestep order, as extract_mimic does.
        text_counts.append(len(episode_text))
        for idx, rid in episode_text:
            text_timesteps.append(idx)
            text_values.append([(rid + k) % 11 for k in range(TEXT_TOKEN_LEN)])
            text_masks_sparse.append([1.0] * TEXT_TOKEN_LEN)
            text_embeddings.append([rid + 0.01 * k for k in range(EMBED_DIM)])

    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(text_counts)

    return dict(
        val_numeric_indicators=numeric_ind,
        val_numeric_values=numeric_vals,
        val_categorical_indicators=categorical_ind,
        val_categorical_values=categorical_vals,
        val_ordinal_indicators=ordinal_ind,
        val_ordinal_values=ordinal_vals,
        val_multilabel_indicators=multilabel_ind,
        val_multilabel_values=multilabel_vals,
        val_text_indicators=text_ind,
        val_times=val_times,
        val_masks=val_masks,
        val_text_offsets=[offsets],
        val_text_values=[np.array(text_values, dtype=np.int64).reshape(-1, TEXT_TOKEN_LEN)],
        val_text_masks=[np.array(text_masks_sparse, dtype=np.float32).reshape(-1, TEXT_TOKEN_LEN)],
        val_text_timesteps=[np.array(text_timesteps, dtype=np.int32)],
        val_text_embeddings=[np.array(text_embeddings, dtype=np.float32).reshape(-1, EMBED_DIM)],
        text_embed_dim=EMBED_DIM,
        event_indicators=event_ind,
        event_times=event_times,
        event_masks=event_masks,
        static_data=static,
        mortality=mortality,
        length_of_stay=length_of_stay,
        phenotype=phenotype,
        max_ts_len=ts_len,
        text_token_len=[TEXT_TOKEN_LEN],
        max_history_len_steps=max_history_len_steps,
    )


def _timestep_tensors(collated, batch_size, n_timesteps):
    """Collect every tensor in a collated batch with shape exactly (batch_size, n_timesteps).

    The MixedTensorDataset nesting is incidental to what these tests check, so walk it rather
    than hard-coding key paths.
    """
    found = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif torch.is_tensor(node) and node.shape == (batch_size, n_timesteps):
            found.append(node)

    walk(collated)
    return found


def _assert_items_equal(a, b, context):
    """Assert two __getitem__ outputs match exactly."""
    assert a.keys() == b.keys(), context
    for key in a:
        left, right = a[key], b[key]
        if isinstance(left, list):
            assert len(left) == len(right), f'{context}: {key} length'
            for f, (x, y) in enumerate(zip(left, right)):
                # Text embeddings come out sparse, as (timestep_index, values) per feature.
                if isinstance(x, tuple):
                    assert len(x) == len(y), f'{context}: {key}[{f}] arity'
                    for part, (u, v) in enumerate(zip(x, y)):
                        assert u.shape == v.shape, \
                            f'{context}: {key}[{f}][{part}] shape {u.shape} != {v.shape}'
                        assert torch.equal(u, v), f'{context}: {key}[{f}][{part}] values'
                    continue
                assert x.shape == y.shape, f'{context}: {key}[{f}] shape {x.shape} != {y.shape}'
                assert torch.equal(x, y), f'{context}: {key}[{f}] values'
        else:
            assert left.shape == right.shape, \
                f'{context}: {key} shape {left.shape} != {right.shape}'
            assert torch.equal(left, right), f'{context}: {key} values'


def test_crop_matches_native_extraction():
    """Cropping a large extraction reproduces a natively-small one, episode for episode."""
    big = MixedDataset(
        **_build_arrays(EPISODE_SPECS, H_BIG, E_BIG),
        history_len_steps=H_SMALL,
        episode_len_steps=E_SMALL,
    )
    small = MixedDataset(**_build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL))

    assert big.ts_len == H_SMALL + E_SMALL
    assert small.ts_len == H_SMALL + E_SMALL
    assert len(big) == len(small) == len(EPISODE_SPECS)

    for i in range(len(EPISODE_SPECS)):
        _assert_items_equal(big[i], small[i], f'episode {i}')


def test_crop_history_only_matches_native_extraction():
    """Capping history alone is equivalent to extracting with a smaller MAX_HISTORY_LEN_STEPS."""
    big = MixedDataset(
        **_build_arrays(EPISODE_SPECS, H_BIG, E_BIG),
        history_len_steps=H_SMALL,
    )
    small = MixedDataset(**_build_arrays(EPISODE_SPECS, H_SMALL, E_BIG))
    for i in range(len(EPISODE_SPECS)):
        _assert_items_equal(big[i], small[i], f'episode {i}')


def test_crop_episode_only_matches_native_extraction():
    """Capping in-stay length alone is equivalent to a smaller MAX_EPISODE_LEN_STEPS."""
    big = MixedDataset(
        **_build_arrays(EPISODE_SPECS, H_BIG, E_BIG),
        episode_len_steps=E_SMALL,
    )
    small = MixedDataset(**_build_arrays(EPISODE_SPECS, H_BIG, E_SMALL))
    for i in range(len(EPISODE_SPECS)):
        _assert_items_equal(big[i], small[i], f'episode {i}')


def test_zero_history_drops_all_history():
    """history_len_steps=0 leaves only in-stay timesteps."""
    ds = MixedDataset(**_build_arrays(EPISODE_SPECS, H_BIG, E_BIG), history_len_steps=0)
    assert ds.ts_len == E_BIG
    native = MixedDataset(**_build_arrays(EPISODE_SPECS, 0, E_BIG))
    for i in range(len(EPISODE_SPECS)):
        _assert_items_equal(ds[i], native[i], f'episode {i}')


def test_no_crop_is_identity():
    """Omitting both caps returns the full extracted window."""
    kwargs = _build_arrays(EPISODE_SPECS, H_BIG, E_BIG)
    ds = MixedDataset(**kwargs)
    assert ds.ts_len == H_BIG + E_BIG
    assert not ds.is_cropped
    item = ds[1]
    assert item['val_masks'].shape == (H_BIG + E_BIG,)
    assert torch.equal(
        item['val_times'], torch.from_numpy(kwargs['val_times'][1].copy())
    )


def test_lengthening_is_rejected():
    """Requests beyond what was extracted fail loudly rather than silently padding."""
    kwargs = _build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL)
    for bad in ({'history_len_steps': H_SMALL + 1}, {'episode_len_steps': E_SMALL + 1}):
        try:
            MixedDataset(**kwargs, **bad)
        except ValueError as exc:
            assert 'exceeds' in str(exc), str(exc)
        else:
            raise AssertionError(f'expected ValueError for {bad}')

    try:
        MixedDataset(**kwargs, history_len_steps=-1)
    except ValueError as exc:
        assert 'non-negative' in str(exc), str(exc)
    else:
        raise AssertionError('expected ValueError for negative length')


def test_collate_masks_post_crop_history_region():
    """Dropping history must clear the cropped history region, not the stored one."""
    ds = MixedDataset(
        **_build_arrays(EPISODE_SPECS, H_BIG, E_BIG),
        history_len_steps=H_SMALL,
        episode_len_steps=E_SMALL,
    )
    batch = [ds[i] for i in range(len(EPISODE_SPECS))]
    masked = collate_tensorized(
        batch, use_historical_nontext_records=False, use_historical_text_records=False,
        history_len_steps=ds.history_len_steps
    )
    unmasked = collate_tensorized(
        [ds[i] for i in range(len(EPISODE_SPECS))],
        history_len_steps=ds.history_len_steps
    )

    masked_tensors = _timestep_tensors(masked, len(EPISODE_SPECS), ds.ts_len)
    unmasked_tensors = _timestep_tensors(unmasked, len(EPISODE_SPECS), ds.ts_len)
    assert masked_tensors, 'no per-timestep tensors found in collated batch'

    # The mask tensors are exactly those the collate function zeroed over the history region.
    mask_pairs = [
        (m, u) for m, u in zip(masked_tensors, unmasked_tensors)
        if torch.any(u[:, :H_SMALL] != 0.0) and torch.all(m[:, :H_SMALL] == 0.0)
    ]
    # Only the value stream still has a history region to mask: the event stream is sliced to
    # in-stay records at collation, so the history switches have nothing left to zero there.
    assert len(mask_pairs) >= 1, \
        f'expected the val mask to be zeroed, found {len(mask_pairs)}'
    for m, u in mask_pairs:
        assert torch.equal(m[:, H_SMALL:], u[:, H_SMALL:]), 'in-stay region altered'

    for key in ('indicators', 'times', 'masks'):
        assert masked['event_data'][key].shape[1] == E_SMALL, \
            f"event_data['{key}'] should be in-stay only, got {masked['event_data'][key].shape[1]}"
    assert torch.all(masked['event_data']['masks'][:, 0] == 1.0), \
        'event index 0 must be an observed record, not padding'


def test_text_counts_exclude_cropped_entries():
    """Text-balanced sampling must count only text inside the crop window."""
    kwargs = _build_arrays(EPISODE_SPECS, H_BIG, E_BIG)
    full = MixedDataset(**kwargs)
    cropped = MixedDataset(
        **kwargs, history_len_steps=H_SMALL, episode_len_steps=E_SMALL
    )
    native = MixedDataset(**_build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL))

    full_counts = get_text_counts_from_dataset_vectorized(full)
    cropped_counts = get_text_counts_from_dataset_vectorized(cropped)
    native_counts = get_text_counts_from_dataset_vectorized(native)

    assert np.array_equal(cropped_counts, native_counts), \
        f'{cropped_counts} != {native_counts}'
    assert cropped_counts.sum() < full_counts.sum(), \
        'test data does not exercise dropped text entries'
    # Cross-check against what __getitem__ actually materialises. The sparse form carries one
    # row per surviving note, so the row count is the count -- no need to scan for zeros, and
    # unlike the dense form it cannot be confused by a genuinely all-zero embedding.
    for i in range(len(EPISODE_SPECS)):
        timesteps, values = cropped[i]['val_text_embeddings'][0]
        assert timesteps.shape[0] == values.shape[0], f'episode {i}: index/row mismatch'
        kept = int(timesteps.shape[0])
        assert kept == cropped_counts[i], f'episode {i}: {kept} != {cropped_counts[i]}'


def test_save_load_roundtrip_preserves_layout_and_crops():
    """metadata.pkl carries the history split so loaders can crop without the dataset config."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'train')
        save_dataset(MixedDataset(**_build_arrays(EPISODE_SPECS, H_BIG, E_BIG)), path)

        with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        assert metadata['max_history_len_steps'] == H_BIG
        assert metadata['max_ts_len'] == H_BIG + E_BIG

        loaded = load_dataset(
            path, history_len_steps=H_SMALL, episode_len_steps=E_SMALL
        )
        native = MixedDataset(**_build_arrays(EPISODE_SPECS, H_SMALL, E_SMALL))
        for i in range(len(EPISODE_SPECS)):
            _assert_items_equal(loaded[i], native[i], f'episode {i}')

        # Legacy datasets predate the recorded split: cropping must refuse to guess.
        del metadata['max_history_len_steps']
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        try:
            load_dataset(path, history_len_steps=H_SMALL)
        except ValueError as exc:
            assert 'max_history_len_steps' in str(exc), str(exc)
        else:
            raise AssertionError('expected ValueError for unknown extracted layout')

        # ...but an explicit extracted_history_len_steps recovers the same result.
        recovered = load_dataset(
            path, history_len_steps=H_SMALL, episode_len_steps=E_SMALL,
            extracted_history_len_steps=H_BIG
        )
        for i in range(len(EPISODE_SPECS)):
            _assert_items_equal(recovered[i], native[i], f'episode {i}')

        # Loading a legacy dataset without a crop stays backward-compatible.
        assert load_dataset(path).ts_len == H_BIG + E_BIG
    finally:
        shutil.rmtree(tmp)


def test_prepare_dataloaders_applies_crop():
    """End-to-end: prepare_dataloaders crops and masks against the cropped frame."""
    tmp = tempfile.mkdtemp()
    try:
        for partition in ('train', 'test'):
            save_dataset(
                MixedDataset(**_build_arrays(EPISODE_SPECS, H_BIG, E_BIG)),
                os.path.join(tmp, partition),
            )

        loaders = prepare_dataloaders(
            tmp,
            batch_size=len(EPISODE_SPECS),
            num_workers=0,
            pin_memory=False,
            history_len_steps=H_SMALL,
            episode_len_steps=E_SMALL,
        )
        assert len(loaders) == 2, f'expected train and test loaders, got {len(loaders)}'

        for loader in loaders:
            assert loader.dataset.ts_len == H_SMALL + E_SMALL
            assert loader.dataset.history_len_steps == H_SMALL
            batch = next(iter(loader))
            n_batch = len(EPISODE_SPECS)
            assert _timestep_tensors(batch, n_batch, H_SMALL + E_SMALL), \
                'no tensors on the cropped timestep axis'
            assert not _timestep_tensors(batch, n_batch, H_BIG + E_BIG), \
                'batch still carries the uncropped timestep axis'

        # The collate partial must mask the cropped history region, not the extracted one.
        no_history = prepare_dataloaders(
            tmp,
            batch_size=len(EPISODE_SPECS),
            num_workers=0,
            pin_memory=False,
            use_historical_nontext_records=False,
            use_historical_text_records=False,
            history_len_steps=H_SMALL,
            episode_len_steps=E_SMALL,
        )[0]
        batch = next(iter(no_history))
        masks = _timestep_tensors(batch, len(EPISODE_SPECS), H_SMALL + E_SMALL)
        masked_leading = [t for t in masks if torch.all(t[:, :H_SMALL] == 0.0)]
        assert len(masked_leading) >= 1, \
            'expected the val mask to be zeroed over the cropped history region'
        # The event stream carries no history region at all -- it is sliced to in-stay records.
        assert batch['event_data']['masks'].shape[1] == E_SMALL
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\n{len(tests)} tests passed')
