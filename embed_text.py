"""
Pre-embed text features with a frozen LLM and write embeddings to dataset
directories.

Scans all fold/partition directories under --data-dir for extracted datasets,
collects non-empty text token sequences, batches them through the LLM, and
writes the resulting embedding arrays alongside the existing sparse text
storage. Episode IDs are used to deduplicate across folds so that each unique
text entry is embedded at most once.

The LLM is loaded with ``device_map="auto"`` so that HuggingFace
automatically distributes model layers across all visible GPUs (pipeline
parallelism).  To control which GPUs are used, set the
``CUDA_VISIBLE_DEVICES`` environment variable before running this script.
The model is loaded in bfloat16 to halve memory requirements.

After running this script, each partition directory will contain:
    val_text_embeddings_{i}.npy  -- (n_non_empty, embed_dim) float32
    metadata.pkl                 -- updated with 'text_embed_dim'

Usage:
    python embed_text.py --data-dir /path/to/data [--batch-size 64]
"""

import argparse
import gc
import numpy as np
import os
import pickle
import re
import torch

from typing import Dict, List, Optional, Tuple

from TransEHR2.modules import GradientTraceableLLM


def get_fold_names(data_dir: str) -> List[str]:
    """Get all fold directory names sorted numerically."""
    fold_names = []
    for item in os.listdir(data_dir):
        if re.match(r'fold\d+', item) \
                and os.path.isdir(os.path.join(data_dir, item)):
            fold_names.append(item)
    fold_names.sort()
    return fold_names


def get_partition_dirs_for_folds(
    data_dir: str,
    fold_names: List[str]
) -> List[str]:
    """Get partition directories for the given folds.

    Args:
        data_dir: Root data directory containing fold subdirectories.
        fold_names: List of fold names to include.

    Returns:
        List of partition directory paths (fold*/train/, fold*/val/,
        fold*/test/) that contain a metadata.pkl file.
    """
    partition_dirs = []
    for fold in fold_names:
        for split in ['train', 'val', 'test']:
            part_dir = os.path.join(data_dir, fold, split)
            if os.path.isdir(part_dir):
                if os.path.exists(
                    os.path.join(part_dir, 'metadata.pkl')
                ):
                    partition_dirs.append(part_dir)
    return partition_dirs


def load_partition_text(
    part_dir: str
) -> Tuple[
    dict, List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray], int
]:
    """Load metadata and sparse text arrays from a partition directory.

    Returns:
        Tuple of (metadata, offsets, values, masks, timesteps,
        n_text_feats)
    """
    with open(os.path.join(part_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    n_text_feats = metadata['n_text_feats']

    offsets = []
    values = []
    masks = []
    timesteps = []
    for i in range(n_text_feats):
        offsets.append(np.load(
            os.path.join(part_dir, f'val_text_offsets_{i}.npy'),
            mmap_mode='r'
        ))
        values.append(np.load(
            os.path.join(part_dir, f'val_text_values_{i}.npy'),
            mmap_mode='r'
        ))
        masks.append(np.load(
            os.path.join(part_dir, f'val_text_masks_{i}.npy'),
            mmap_mode='r'
        ))
        timesteps.append(np.load(
            os.path.join(part_dir, f'val_text_timesteps_{i}.npy'),
            mmap_mode='r'
        ))

    return metadata, offsets, values, masks, timesteps, n_text_feats


def load_episode_ids(part_dir: str) -> Optional[List]:
    """Load the patient-episode ids for a partition, in extracted-array row order.

    `extract_mimic` writes the ids beside the partition directory rather than inside it --
    `{fold}/{partition}_ids.pkl` against `{fold}/{partition}/` -- so that is where this looks
    first. The partition directory is still checked, for datasets written before that layout.

    These ids are what makes deduplication possible: the folds re-partition one set of
    patients, so an episode recurs in every fold and its text needs embedding once. Without
    them the cache key falls back to the partition path and nothing can ever hit.

    Returns:
        The ids, or None if no file was found.
    """
    split = os.path.basename(part_dir)
    candidates = [
        os.path.join(os.path.dirname(part_dir), f'{split}_ids.pkl'),
        os.path.join(part_dir, f'{split}_ids.pkl'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    for fname in sorted(os.listdir(part_dir)):
        if fname.endswith('_ids.pkl'):
            with open(os.path.join(part_dir, fname), 'rb') as f:
                return pickle.load(f)
    return None


def length_sorted_batches(token_array, mask_array, flat_indices, batch_size):
    """Yield batches ordered by real sequence length, with the padding tail trimmed.

    Sequences are stored padded to MAX_TOKEN_LENGTH. A batch drawn in storage order almost
    always contains one long note, so trimming alone would save nothing; sorting by real length
    first makes each batch nearly uniform, and the trim then costs each batch its own longest
    sequence rather than the global maximum. Attention is quadratic in that length.

    Padding is on the right -- `LLMTextProcessor` refuses a tokenizer that pads left, because
    CLS pooling reads position 0 -- so dropping the tail beyond a batch's longest real sequence
    removes only padding, and the embeddings are unchanged.

    Args:
        token_array: (n, token_len) token ids.
        mask_array: (n, token_len) attention masks, nonzero on real tokens.
        flat_indices: (n,) destination index of each row in the partition's embedding array.
        batch_size: Rows per batch.

    Yields:
        (indices, tokens, masks) per batch, where `indices` says where each row's embedding
        belongs. Rows are permuted, so the caller must scatter by `indices` rather than assume
        input order.
    """
    lengths = np.asarray(mask_array).sum(axis=1)
    order = np.argsort(lengths, kind='stable')
    token_array = np.asarray(token_array)[order]
    mask_array = np.asarray(mask_array)[order]
    flat_indices = np.asarray(flat_indices)[order]

    for start in range(0, len(flat_indices), batch_size):
        end = min(start + batch_size, len(flat_indices))
        masks = mask_array[start:end]
        # At least one column, so an all-empty batch still has a shape the model accepts.
        keep = max(int(masks.sum(axis=1).max()), 1)
        yield flat_indices[start:end], token_array[start:end, :keep], masks[:, :keep]


@torch.no_grad()
def embed_batch(
    llm: GradientTraceableLLM,
    token_ids_batch: np.ndarray,
    mask_batch: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of token sequences through the LLM.

    Args:
        llm: The GradientTraceableLLM instance.
        token_ids_batch: (batch_size, token_len) int64 array of token
            IDs.
        mask_batch: (batch_size, token_len) int64 array of attention
            masks.
        device: Torch device for input tensors (should match the
            device of the model's first parameter).

    Returns:
        (batch_size, embed_dim) float32 numpy array of embeddings.
    """
    ids_tensor = torch.from_numpy(token_ids_batch).long().to(device)
    mask_tensor = torch.from_numpy(mask_batch).long().to(device)
    embeddings = llm(
        ids_tensor, trace_grads=False, attention_mask=mask_tensor
    )
    return embeddings.cpu().float().numpy()


def process_partition(
    part_dir: str,
    llm: GradientTraceableLLM,
    device: torch.device,
    batch_size: int,
    embedding_cache: Dict[str, Dict[int, np.ndarray]],
    embed_dim: int,
) -> int:
    """Process one partition directory: embed all text and write results.

    Args:
        part_dir: Path to the partition directory.
        llm: The loaded LLM module.
        device: Torch device for input tensors.
        batch_size: Batch size for LLM inference.
        embedding_cache: Dict mapping cache_key ->
            {local_idx -> embedding}.  Used for deduplication across
            partitions within the same fold.
        embed_dim: Dimensionality of the LLM embeddings.

    Returns:
        Number of new embeddings computed (not from cache).
    """
    metadata, offsets, values, masks, timesteps, n_text_feats = \
        load_partition_text(part_dir)
    episode_ids = load_episode_ids(part_dir)
    if episode_ids is None:
        # Without ids the cache key falls back to the partition path, which is unique, so every
        # text is re-embedded in every fold. That is six times the work and it is invisible in
        # the output, so say so rather than quietly running long.
        print(f"  WARNING: no episode ids beside {part_dir}. Deduplication across folds is "
              f"off, so this partition re-embeds text other folds already did.")
    elif len(episode_ids) != offsets[0].shape[0] - 1:
        raise ValueError(
            f'{part_dir}: {offsets[0].shape[0] - 1} episodes in the arrays but '
            f'{len(episode_ids)} ids beside them. Caching on mismatched ids would attach one '
            f"episode's embeddings to another."
        )
    # CSR format: n+1 offsets for n episodes
    n_episodes = offsets[0].shape[0] - 1

    new_embeddings_count = 0

    for f in range(n_text_feats):
        # Total non-empty entries for this feature
        n_non_empty = int(offsets[f][-1])
        if n_non_empty == 0:
            np.save(
                os.path.join(
                    part_dir, f'val_text_embeddings_{f}.npy'
                ),
                np.zeros((0, embed_dim), dtype=np.float32)
            )
            continue

        all_embeddings = np.zeros(
            (n_non_empty, embed_dim), dtype=np.float32
        )

        need_embed_indices = []
        need_embed_tokens = []
        need_embed_masks = []

        for ep_idx in range(n_episodes):
            start = int(offsets[f][ep_idx])
            end = int(offsets[f][ep_idx + 1])
            if start == end:
                continue

            ep_id = (
                str(episode_ids[ep_idx])
                if episode_ids is not None
                else f"{part_dir}_{ep_idx}"
            )
            cache_key = f"{ep_id}_f{f}"

            if cache_key in embedding_cache:
                cached = embedding_cache[cache_key]
                n_entries = end - start
                for local_i in range(n_entries):
                    if local_i in cached:
                        all_embeddings[start + local_i] = \
                            cached[local_i]
            else:
                for local_i in range(end - start):
                    flat_idx = start + local_i
                    need_embed_indices.append(flat_idx)
                    need_embed_tokens.append(
                        np.array(values[f][flat_idx])
                    )
                    need_embed_masks.append(
                        np.array(masks[f][flat_idx])
                    )

        if need_embed_indices:
            token_array = np.stack(need_embed_tokens, axis=0)
            mask_array = np.stack(need_embed_masks, axis=0)

            for batch_indices, batch_tokens, batch_masks in length_sorted_batches(
                token_array, mask_array, need_embed_indices, batch_size
            ):
                batch_embeds = embed_batch(
                    llm, batch_tokens, batch_masks, device
                )
                for i, flat_idx in enumerate(batch_indices):
                    all_embeddings[flat_idx] = batch_embeds[i]

            new_embeddings_count += len(need_embed_indices)

        # Populate cache for future partitions within this fold
        for ep_idx in range(n_episodes):
            start = int(offsets[f][ep_idx])
            end = int(offsets[f][ep_idx + 1])
            if start == end:
                continue

            ep_id = (
                str(episode_ids[ep_idx])
                if episode_ids is not None
                else f"{part_dir}_{ep_idx}"
            )
            cache_key = f"{ep_id}_f{f}"

            if cache_key not in embedding_cache:
                cached = {}
                for local_i in range(end - start):
                    cached[local_i] = \
                        all_embeddings[start + local_i].copy()
                embedding_cache[cache_key] = cached

        np.save(
            os.path.join(
                part_dir, f'val_text_embeddings_{f}.npy'
            ),
            all_embeddings
        )
        print(
            f"    Feature {f}: wrote {n_non_empty} embeddings "
            f"({len(need_embed_indices)} new, "
            f"{n_non_empty - len(need_embed_indices)} cached)"
        )

        del all_embeddings, need_embed_tokens, need_embed_masks
        if 'token_array' in dir():
            del token_array, mask_array

    metadata['text_embed_dim'] = embed_dim
    with open(os.path.join(part_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return new_embeddings_count


def main():
    parser = argparse.ArgumentParser(
        description='Pre-embed text features with a frozen LLM'
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Root data directory containing fold subdirectories'
    )
    parser.add_argument(
        '--llm-name', type=str, default=None,
        help='HuggingFace model name for the LLM. '
             'Defaults to the LLM_NAME constant in '
             'TransEHR2.constants.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for LLM inference (default: 64)'
    )
    args = parser.parse_args()

    all_fold_names = get_fold_names(args.data_dir)
    if not all_fold_names:
        print(f"No fold directories found under {args.data_dir}")
        return

    all_partition_dirs = get_partition_dirs_for_folds(
        args.data_dir, all_fold_names
    )
    print(
        f"Found {len(all_fold_names)} folds, "
        f"{len(all_partition_dirs)} partition directories, "
        f"batch_size={args.batch_size}"
    )

    if not all_partition_dirs:
        print("No partition directories to process.")
        return

    # Load LLM with device_map="auto" to distribute across all
    # visible GPUs, and bfloat16 to halve memory.
    print("Loading LLM with device_map='auto'...")
    llm_kwargs = {}
    if args.llm_name:
        llm_kwargs['model_name'] = args.llm_name
    llm = GradientTraceableLLM(
        use_gradient_checkpointing=False,
        device_map='auto',
        dtype=torch.bfloat16,
        **llm_kwargs,
    )
    llm.eval()
    embed_dim = llm.model.config.hidden_size
    # Input tensors must be on the device of the model's first
    # parameter (the embedding layer).
    input_device = next(llm.model.parameters()).device
    print(
        f"LLM loaded: {llm.model.config._name_or_path}, "
        f"embed_dim={embed_dim}, input_device={input_device}"
    )

    embedding_cache: Dict[str, Dict[int, np.ndarray]] = {}
    total_new = 0

    for part_dir in all_partition_dirs:
        print(f"Processing: {part_dir}")
        new_count = process_partition(
            part_dir, llm, input_device, args.batch_size,
            embedding_cache, embed_dim
        )
        total_new += new_count
        print(f"  {new_count} new embeddings computed")

        torch.cuda.empty_cache()

    print(f"Done. Total new embeddings: {total_new}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
