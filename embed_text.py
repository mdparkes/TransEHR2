"""
Pre-embed text features with a frozen LLM and write embeddings to dataset directories.

Scans all fold/partition directories under --data-dir for extracted datasets,
collects non-empty text token sequences, batches them through the LLM, and
writes the resulting embedding arrays alongside the existing sparse text
storage. Episode IDs are used to deduplicate across folds so that each unique
text entry is embedded at most once.

Uses HuggingFace Accelerate for distributed inference across multiple GPUs.
When running multi-GPU, folds are assigned round-robin to ranks so that each
rank processes disjoint directories (no write conflicts).  Within-fold
deduplication (train/val/test sharing episodes) is preserved because all
partitions of the same fold are processed by the same rank.

Also works on a single GPU without ``accelerate launch``.

After running this script, each partition directory will contain:
    val_text_embeddings_{i}.npy  -- (n_non_empty, embed_dim) float32
    metadata.pkl                 -- updated with 'text_embed_dim'

Usage (single GPU):
    python embed_text.py --data-dir /path/to/data [--llm-name meta-llama/Llama-3.2-1B] \
        [--batch-size 64]

Usage (multi-GPU):
    accelerate launch embed_text.py --data-dir /path/to/data \
        [--llm-name meta-llama/Llama-3.2-1B] [--batch-size 64]
"""

import argparse
import gc
import numpy as np
import os
import pickle
import re
import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from typing import Dict, List, Optional, Tuple

from TransEHR2.modules import GradientTraceableLLM


def initialize_inference_accelerator() -> Accelerator:
    """Initialize an Accelerator for inference.

    Unlike the training accelerator, this does not enforce a specific
    distributed type, so the script works both via ``accelerate launch``
    (multi-GPU) and plain ``python`` (single GPU).

    Returns:
        Accelerator: Configured Accelerator instance.
    """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(kwargs_handlers=[ddp_kwargs])


def get_fold_names(data_dir: str) -> List[str]:
    """Get all fold directory names sorted numerically."""
    fold_names = []
    for item in os.listdir(data_dir):
        if re.match(r'fold\d+', item) and os.path.isdir(os.path.join(data_dir, item)):
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
                if os.path.exists(os.path.join(part_dir, 'metadata.pkl')):
                    partition_dirs.append(part_dir)
    return partition_dirs


def load_partition_text(
    part_dir: str
) -> Tuple[dict, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """Load metadata and sparse text arrays from a partition directory.

    Returns:
        Tuple of (metadata, offsets, values, masks, timesteps, n_text_feats)
    """
    with open(os.path.join(part_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    n_text_feats = metadata['n_text_feats']

    offsets = []
    values = []
    masks = []
    timesteps = []
    for i in range(n_text_feats):
        offsets.append(np.load(os.path.join(part_dir, f'val_text_offsets_{i}.npy'), mmap_mode='r'))
        values.append(np.load(os.path.join(part_dir, f'val_text_values_{i}.npy'), mmap_mode='r'))
        masks.append(np.load(os.path.join(part_dir, f'val_text_masks_{i}.npy'), mmap_mode='r'))
        timesteps.append(np.load(os.path.join(part_dir, f'val_text_timesteps_{i}.npy'), mmap_mode='r'))

    return metadata, offsets, values, masks, timesteps, n_text_feats


def load_episode_ids(part_dir: str) -> Optional[List]:
    """Load episode IDs from a partition directory.

    Tries common naming conventions: train_ids.pkl, val_ids.pkl, test_ids.pkl.
    Returns None if no ID file is found.
    """
    split = os.path.basename(part_dir)
    ids_path = os.path.join(part_dir, f'{split}_ids.pkl')
    if os.path.exists(ids_path):
        with open(ids_path, 'rb') as f:
            return pickle.load(f)
    # Fallback: look for any *_ids.pkl
    for fname in os.listdir(part_dir):
        if fname.endswith('_ids.pkl'):
            with open(os.path.join(part_dir, fname), 'rb') as f:
                return pickle.load(f)
    return None


@torch.no_grad()
def embed_batch(
    llm: GradientTraceableLLM,
    token_ids_batch: np.ndarray,
    mask_batch: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of token sequences through the LLM.

    Args:
        llm: The GradientTraceableLLM instance (possibly wrapped by
            Accelerate).
        token_ids_batch: (batch_size, token_len) int64 array of token IDs.
        mask_batch: (batch_size, token_len) int64 array of attention masks.
        device: Torch device to run on.

    Returns:
        (batch_size, embed_dim) float32 numpy array of embeddings.
    """
    ids_tensor = torch.from_numpy(token_ids_batch).long().to(device)
    mask_tensor = torch.from_numpy(mask_batch).long().to(device)
    embeddings = llm(ids_tensor, trace_grads=False, attention_mask=mask_tensor)
    return embeddings.cpu().float().numpy()


def process_partition(
    part_dir: str,
    llm: GradientTraceableLLM,
    device: torch.device,
    batch_size: int,
    embedding_cache: Dict[str, Dict[int, np.ndarray]],
    embed_dim: int,
) -> int:
    """Process one partition directory, embedding all text and writing results.

    Args:
        part_dir: Path to the partition directory.
        llm: The loaded LLM module (possibly wrapped by Accelerate).
        device: Torch device.
        batch_size: Batch size for LLM inference.
        embedding_cache: Dict mapping cache_key -> {local_idx -> embedding}.
            Used for deduplication across partitions within the same fold.
        embed_dim: Dimensionality of the LLM embeddings.

    Returns:
        Number of new embeddings computed (not from cache).
    """
    metadata, offsets, values, masks, timesteps, n_text_feats = load_partition_text(part_dir)
    episode_ids = load_episode_ids(part_dir)
    n_episodes = offsets[0].shape[0] - 1  # CSR format: n+1 offsets for n episodes

    new_embeddings_count = 0

    for f in range(n_text_feats):
        n_non_empty = int(offsets[f][-1])  # Total non-empty entries for this feature
        if n_non_empty == 0:
            # No text entries for this feature, write empty array
            np.save(
                os.path.join(part_dir, f'val_text_embeddings_{f}.npy'),
                np.zeros((0, embed_dim), dtype=np.float32)
            )
            continue

        # Allocate output array
        all_embeddings = np.zeros((n_non_empty, embed_dim), dtype=np.float32)

        # Collect entries that need embedding vs those we can copy from cache
        need_embed_indices = []  # Indices into the flat sparse array that need LLM
        need_embed_tokens = []
        need_embed_masks = []

        for ep_idx in range(n_episodes):
            start = int(offsets[f][ep_idx])
            end = int(offsets[f][ep_idx + 1])
            if start == end:
                continue

            ep_id = str(episode_ids[ep_idx]) if episode_ids is not None else f"{part_dir}_{ep_idx}"
            cache_key = f"{ep_id}_f{f}"

            if cache_key in embedding_cache:
                # Copy from cache
                cached = embedding_cache[cache_key]
                n_entries = end - start
                for local_i in range(n_entries):
                    if local_i in cached:
                        all_embeddings[start + local_i] = cached[local_i]
            else:
                # Need to embed these
                for local_i in range(end - start):
                    flat_idx = start + local_i
                    need_embed_indices.append(flat_idx)
                    need_embed_tokens.append(np.array(values[f][flat_idx]))
                    need_embed_masks.append(np.array(masks[f][flat_idx]))

        # Batch embed all needed entries
        if need_embed_indices:
            token_array = np.stack(need_embed_tokens, axis=0)
            mask_array = np.stack(need_embed_masks, axis=0)

            for batch_start in range(0, len(need_embed_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(need_embed_indices))
                batch_tokens = token_array[batch_start:batch_end]
                batch_masks = mask_array[batch_start:batch_end]
                batch_embeds = embed_batch(llm, batch_tokens, batch_masks, device)

                for i, flat_idx in enumerate(need_embed_indices[batch_start:batch_end]):
                    all_embeddings[flat_idx] = batch_embeds[i]

            new_embeddings_count += len(need_embed_indices)

        # Populate cache for future partitions within this fold
        for ep_idx in range(n_episodes):
            start = int(offsets[f][ep_idx])
            end = int(offsets[f][ep_idx + 1])
            if start == end:
                continue

            ep_id = str(episode_ids[ep_idx]) if episode_ids is not None else f"{part_dir}_{ep_idx}"
            cache_key = f"{ep_id}_f{f}"

            if cache_key not in embedding_cache:
                cached = {}
                for local_i in range(end - start):
                    cached[local_i] = all_embeddings[start + local_i].copy()
                embedding_cache[cache_key] = cached

        # Write embeddings
        np.save(
            os.path.join(part_dir, f'val_text_embeddings_{f}.npy'),
            all_embeddings
        )
        print(f"    Feature {f}: wrote {n_non_empty} embeddings "
              f"({len(need_embed_indices)} new, {n_non_empty - len(need_embed_indices)} cached)")

        # Clear references to help GC
        del all_embeddings, need_embed_tokens, need_embed_masks
        if 'token_array' in dir():
            del token_array, mask_array

    # Update metadata with text_embed_dim
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
             'Defaults to the LLM_NAME constant in TransEHR2.constants.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for LLM inference (default: 64)'
    )
    args = parser.parse_args()

    # ---- Initialize Accelerator ----
    accelerator = initialize_inference_accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}")
        print(f"Number of processes: {world_size}")

    # Discover folds and assign to ranks round-robin
    all_fold_names = get_fold_names(args.data_dir)
    if not all_fold_names:
        if is_main:
            print(f"No fold directories found under {args.data_dir}")
        return

    my_folds = [
        all_fold_names[i]
        for i in range(rank, len(all_fold_names), world_size)
    ]
    my_partition_dirs = get_partition_dirs_for_folds(
        args.data_dir, my_folds
    )

    print(f"[Rank {rank}] Assigned {len(my_folds)} folds: {my_folds}")
    print(f"[Rank {rank}] {len(my_partition_dirs)} partition directories")

    # Load LLM
    if is_main:
        print("\nLoading LLM...")
    kwargs = {}
    if args.llm_name:
        kwargs['model_name'] = args.llm_name
    llm = GradientTraceableLLM(**kwargs)
    llm.eval()
    embed_dim = llm.model.config.hidden_size

    # Wrap with Accelerate for distributed placement
    llm = accelerator.prepare(llm)

    if is_main:
        unwrapped = accelerator.unwrap_model(llm)
        print(f"LLM loaded: "
              f"{unwrapped.model.config._name_or_path}, "
              f"embed_dim={embed_dim}")

    accelerator.wait_for_everyone()

    # Process partitions assigned to this rank
    embedding_cache: Dict[str, Dict[int, np.ndarray]] = {}
    total_new = 0

    for part_dir in my_partition_dirs:
        print(f"[Rank {rank}] Processing: {part_dir}")
        new_count = process_partition(
            part_dir, llm, device, args.batch_size,
            embedding_cache, embed_dim
        )
        total_new += new_count
        print(f"[Rank {rank}]   {new_count} new embeddings computed")

        # Periodic GPU cache cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Aggregate totals across ranks
    accelerator.wait_for_everyone()
    total_new_tensor = torch.tensor(
        [total_new], device=device, dtype=torch.long
    )
    all_totals = accelerator.gather(total_new_tensor)

    if is_main:
        grand_total = all_totals.sum().item()
        print(f"\nDone. Total new embeddings computed across "
              f"all ranks: {grand_total}")

    # Clean up
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
