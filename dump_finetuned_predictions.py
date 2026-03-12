"""
Extract predictions from finetuned TransEHR2 models for evaluation.

For each cross-validation fold and prediction task (mortality, length_of_stay,
phenotype), loads the finetuned model weights, performs a forward pass on
training, validation (if available), and test data, and saves predictions
and targets to CSV files.

Uses HuggingFace Accelerate for distributed inference across multiple GPUs.
Also works on a single GPU without ``accelerate launch``.

Output files are written to:
    {model_dir}/{experiment_name}/{fold}/{task}/{task}_{split}_finetuned_output.csv

Usage (single GPU):
    python dump_finetuned_predictions.py <dataset_config> <experiment_config> <experiment_name> \
        [--model_dir ./models] [--num_workers 0] [--batch_size 750]

Usage (multi-GPU with config):
    accelerate launch --config_file <accelerate_config.yaml> \
        dump_finetuned_predictions.py <dataset_config> <experiment_config> \
        <experiment_name> [--model_dir ./models] [--num_workers 0] [--batch_size 750]
"""

import argparse
import gc
import math
import numpy as np
import os
import pandas as pd
import re
import torch
import yaml

from accelerate import Accelerator, DistributedDataParallelKwargs
from collections import OrderedDict
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from TransEHR2.constants import TEXT_EMBED_DIM
from TransEHR2.data.preprocessing import load_dataset, collate_tensorized
from TransEHR2.models import MixedClassifier
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder, GradientTraceableLLM
from TransEHR2.utils import get_param_shapes, move_batch_to_device


# ---------------------------------------------------------------------------
# Inlined from TransEHR2.routines_accelerate to avoid pulling in the
# tensorboard dependency that module carries at import time.
# ---------------------------------------------------------------------------

StateDict = OrderedDict[str, Tensor]


def reshape_flattened_state_dict(
    state_dict: StateDict,
    param_shapes: OrderedDict[str, tuple]
) -> StateDict:
    """Reshape flattened FSDP state dict to match expected parameter shapes.

    LLM parameters are intentionally excluded from state dicts since the LLM
    is frozen and always initialised from HuggingFace weights.  This function
    is primarily needed for FSDP, which may flatten parameters.  For DDP,
    parameters retain their original shapes.
    """

    def strip_fsdp_prefix(key: str) -> str:
        for prefix in [
            '_fsdp_wrapped_module.', '_forward_module.', 'module.'
        ]:
            if prefix in key:
                key = key.replace(prefix, '')
        if key.startswith('llm_model.'):
            key = key.replace('llm_model.', 'llm_module.model.')
        return key

    reshaped: StateDict = OrderedDict()

    for key, tensor in state_dict.items():
        clean_key = strip_fsdp_prefix(key)

        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        if clean_key in param_shapes:
            expected_shape = param_shapes[clean_key]
            if tensor.shape != expected_shape:
                expected_numel = int(
                    torch.prod(torch.tensor(expected_shape)).item()
                )
                if tensor.numel() != expected_numel:
                    print(
                        f"ERROR: Cannot reshape {clean_key}: "
                        f"{tensor.numel()} elements vs expected "
                        f"{expected_numel}"
                    )
                    reshaped[clean_key] = tensor.clone()
                else:
                    reshaped[clean_key] = tensor.reshape(
                        expected_shape
                    ).clone()
            else:
                reshaped[clean_key] = tensor.clone()
        else:
            if not (clean_key.startswith('llm_module.')
                    or clean_key.startswith('llm_model.')):
                print(
                    f"Warning: No expected shape for {clean_key}, "
                    f"keeping original shape {tensor.shape}"
                )
            reshaped[clean_key] = tensor.clone()

    return reshaped


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


def get_fold_names(data_dir: str, exclude: Optional[List[str]] = None) -> List[str]:
    """Get cross-validation fold names from directory structure.

    Args:
        data_dir: Path to the directory containing fold subdirectories.
        exclude: List of fold names to exclude from the results.

    Returns:
        Sorted list of fold directory names matching the pattern 'fold\\d+'.
    """
    if exclude is None:
        exclude = []
    fold_names = []
    for item in os.listdir(data_dir):
        if item in exclude:
            continue
        if re.match(r'fold\d+', item) and os.path.isdir(os.path.join(data_dir, item)):
            fold_names.append(item)
    fold_names.sort()
    return fold_names


def get_phenotype_names(fold_dir: str) -> Optional[List[str]]:
    """Read phenotype class names from the phenotyping listfile header.

    Args:
        fold_dir: Path to the fold directory containing listfiles.

    Returns:
        List of phenotype class name strings, or None if the listfile
        is not found.
    """
    listfile = os.path.join(fold_dir, 'phenotyping_test_listfile.csv')
    if os.path.exists(listfile):
        with open(listfile, 'r') as f:
            header = f.readline().strip().split(',')
        # First two columns are 'stay' and 'period_length'
        return header[2:]
    return None


def create_inference_loader(
    fold_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[Optional[DataLoader], int]:
    """Create a DataLoader for one data split, sharded across ranks.

    Data is sharded sequentially so that rank 0 gets the first chunk,
    rank 1 gets the second, etc.  After ``accelerator.gather()``, the
    concatenated predictions are in original dataset order.

    Args:
        fold_dir: Path to the fold directory containing split subdirs.
        split: One of 'train', 'val', or 'test'.
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to use pinned memory for CUDA transfers.
        world_size: Total number of distributed processes.
        rank: Index of the current process.

    Returns:
        Tuple of (DataLoader or None, total_dataset_size).  The loader
        is None only when split is 'val' and the directory is missing.

    Raises:
        FileNotFoundError: If a required split directory ('train' or
            'test') is not found.
    """
    dataset_path = os.path.join(fold_dir, split)
    if not os.path.exists(dataset_path):
        if split == 'val':
            return None, 0
        raise FileNotFoundError(f"'{split}/' not found in {fold_dir}")

    dataset = load_dataset(dataset_path)
    total = len(dataset)

    # Shard sequentially across ranks.  Every rank must receive
    # exactly ``per_rank`` samples so that all ranks produce the same
    # number of batches.  Under FSDP each forward pass triggers
    # all-gather collectives, so an uneven batch count across ranks
    # causes the faster ranks to block at gather while the slower
    # ones are still in the forward pass, eventually timing out.
    per_rank = math.ceil(total / world_size)
    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, total)
    indices = list(range(start_idx, end_idx))

    # Pad with duplicates of the last sample so every rank has
    # exactly per_rank entries.  The duplicates land at the tail of
    # the gathered tensor and are trimmed by actual_n_samples.
    if indices:
        while len(indices) < per_rank:
            indices.append(indices[-1])
    else:
        # Rank beyond dataset size (total < world_size) — fill with
        # index 0 as a dummy; predictions will be trimmed away.
        indices = [0] * per_rank

    subset = Subset(dataset, indices) if world_size > 1 else dataset

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tensorized,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    return loader, total


def build_classifier(
    experiment_config: dict,
    n_val_feats: int,
    tot_val_feat_dim: int,
    n_event_types: int,
    n_static_feats: int,
    num_classes: int,
    use_text: bool
) -> MixedClassifier:
    """Instantiate a MixedClassifier from experiment configuration values.

    Args:
        experiment_config: Parsed YAML experiment configuration.
        n_val_feats: Number of value-associated features (numeric +
            categorical, and optionally text).
        tot_val_feat_dim: Total dimensionality of concatenated value features.
        n_event_types: Number of event feature types.
        n_static_feats: Number of static features.
        num_classes: Number of prediction output classes.
        use_text: Whether to initialise the model with an LLM module.

    Returns:
        An uninitialised (randomly weighted) MixedClassifier instance.
    """
    llm_module = GradientTraceableLLM() if use_text else None

    val_encoder = ValueDataEncoder(
        n_features=n_val_feats,
        feat_dim=tot_val_feat_dim,
        d_model=experiment_config['DISCRIMINATOR_ENCODER_D_MODEL'],
        n_heads=experiment_config['DISCRIMINATOR_ENCODER_N_HEADS'],
        n_encoder_blocks=experiment_config['DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS'],
        dim_feedforward=experiment_config['DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD'],
        dropout=experiment_config['DISCRIMINATOR_ENCODER_DROPOUT'],
        activation=experiment_config['DISCRIMINATOR_ENCODER_ACTIVATION'],
        norm=experiment_config['DISCRIMINATOR_ENCODER_NORM'],
        normalize_before=experiment_config.get('DISCRIMINATOR_ENCODER_NORM_FIRST', True)
    )
    event_encoder = EventDataEncoder(
        num_types=n_event_types,
        d_model=experiment_config['THP_ENCODER_D_MODEL'],
        d_inner=experiment_config['THP_ENCODER_D_INNER'],
        n_layers=experiment_config['THP_ENCODER_N_LAYERS'],
        n_head=experiment_config['THP_ENCODER_N_HEADS'],
        d_k=experiment_config['THP_ENCODER_D_K'],
        d_v=experiment_config['THP_ENCODER_D_V'],
        dropout=experiment_config['THP_ENCODER_DROPOUT'],
        normalize_before=experiment_config.get('THP_ENCODER_NORM_FIRST', True)
    )
    return MixedClassifier(
        event_encoder=event_encoder,
        val_encoder=val_encoder,
        d_event_enc=experiment_config['THP_ENCODER_D_MODEL'],
        d_val_enc=experiment_config['DISCRIMINATOR_ENCODER_D_MODEL'],
        d_statics=n_static_feats,
        num_classes=num_classes,
        aggr=experiment_config['PREDICTOR_AGGREGATION_METHOD'],
        use_text=use_text,
        llm_module=llm_module
    )


def load_finetuned_weights(
    model: MixedClassifier,
    weights_path: str,
    use_text: bool = False
) -> bool:
    """Load finetuned state dict into a MixedClassifier.

    Handles potential FSDP-flattened parameter shapes by reshaping them
    to match the model's expected parameter dimensions.

    Args:
        model: The MixedClassifier to load weights into.
        weights_path: Path to the saved .pt state dict file.
        use_text: Whether the model uses text/LLM features. When False,
            strict loading is used so any key mismatch raises an error
            immediately.

    Returns:
        True if weights were loaded successfully, False if the file was
        not found.
    """
    if not os.path.exists(weights_path):
        return False

    state_dict = torch.load(
        weights_path, map_location='cpu', weights_only=False
    )
    param_shapes = get_param_shapes(model)
    state_dict = reshape_flattened_state_dict(state_dict, param_shapes)

    # When USE_TEXT is False every parameter must be present in the
    # state dict, so use strict=True to surface mismatches immediately.
    # When USE_TEXT is True the frozen LLM params are intentionally
    # absent from the saved state dict.
    strict = not use_text
    result = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if result.missing_keys:
            llm_missing = [k for k in result.missing_keys
                           if k.startswith('llm_module.')]
            other_missing = [k for k in result.missing_keys
                             if not k.startswith('llm_module.')]
            if other_missing:
                print(f"    WARNING: {len(other_missing)} non-LLM keys "
                      f"missing from state dict: {other_missing}")
        if result.unexpected_keys:
            print(f"    WARNING: {len(result.unexpected_keys)} "
                  f"unexpected keys in state dict: "
                  f"{result.unexpected_keys}")

    # Verify no NaN values in loaded parameters
    nan_params = [name for name, p in model.named_parameters()
                  if torch.isnan(p).any()]
    if nan_params:
        print(f"    WARNING: NaN values found in parameters: "
              f"{nan_params}")

    del state_dict
    return True


def install_nan_hooks(model: MixedClassifier) -> List:
    """Install forward hooks that replace NaN encoder output with zeros.

    The ValueDataEncoder uses ``batch_first=True`` with a manually permuted
    input so that each timestep becomes a "batch" processed by PyTorch's
    ``TransformerEncoder``.  When **every** episode in the real batch has
    padding at a given timestep, all key positions for that "batch item"
    are masked, producing ``softmax(-inf, …, -inf) = NaN``.  Those NaN
    values survive the subsequent ``val_enc * mask`` operation because
    ``NaN * 0 = NaN`` in IEEE 754, and then ``torch.sum`` propagates the
    NaN to every prediction in the batch.

    This hook replaces NaN values in each encoder's output with zeros so
    that they are harmlessly absorbed by the padding mask and aggregation.

    Args:
        model: A MixedClassifier whose encoders may produce NaN at
            fully-padded timesteps.

    Returns:
        List of hook handles (call ``.remove()`` on each to uninstall).
    """
    nan_counts: Dict[str, int] = {'val_encoder': 0, 'event_encoder': 0}

    def _make_hook(name: str):
        def hook(module, inp, output):
            if isinstance(output, torch.Tensor):
                n = torch.isnan(output).sum().item()
                if n > 0:
                    nan_counts[name] += n
                    return torch.nan_to_num(output, nan=0.0)
            return output
        return hook

    handles = [
        model.val_encoder.register_forward_hook(_make_hook('val_encoder')),
        model.event_encoder.register_forward_hook(_make_hook('event_encoder')),
    ]
    # Expose the counter dict so callers can inspect it later.
    for h in handles:
        h.nan_counts = nan_counts  # type: ignore[attr-defined]
    return handles


def run_inference(
    model: MixedClassifier,
    loader: DataLoader,
    task: str,
    accelerator: Accelerator,
    actual_n_samples: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run distributed inference and gather predictions across ranks.

    For binary classification tasks (mortality, phenotype), predictions are
    sigmoid-transformed probabilities. For regression (length_of_stay),
    predictions are raw model outputs.

    Each rank processes its sequential shard of the dataset.  After the
    loop, local predictions are padded and gathered so that the main
    process receives all predictions in original dataset order.

    Args:
        model: The finetuned MixedClassifier in eval mode.
        loader: DataLoader for this rank's shard of the data split.
        task: One of 'mortality', 'length_of_stay', or 'phenotype'.
        accelerator: Accelerator instance for distributed inference.
        actual_n_samples: Total number of samples across all ranks
            (used for trimming gathered predictions).

    Returns:
        On the main process: tuple of (predictions, targets) as numpy
        arrays with shape (n_samples, n_outputs).
        On other processes: (None, None).
    """
    model.eval()
    all_preds = []
    all_targs = []
    nan_batches = 0
    is_main = accelerator.is_main_process

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(loader, desc='    Inference', leave=False,
                 disable=not accelerator.is_local_main_process)
        ):
            batch = move_batch_to_device(
                batch, device=accelerator.device
            )
            logits = model(batch)
            targets = batch['targets'][task]

            if task in ('mortality', 'phenotype'):
                preds = torch.sigmoid(logits)
            else:
                preds = logits

            n_nan = torch.isnan(logits).sum().item()
            if n_nan > 0:
                nan_batches += 1

            # Diagnostics on the first batch (main process only)
            if i == 0 and is_main:
                if n_nan > 0:
                    print(f"    WARNING: {n_nan} NaN values in logits "
                          f"(batch 0, shape {tuple(logits.shape)})")
                    print(f"    logits sample: "
                          f"{logits[0].cpu().tolist()}")
                else:
                    print(f"    logits OK (batch 0): "
                          f"min={logits.min().item():.4f}, "
                          f"max={logits.max().item():.4f}")

            all_preds.append(preds.detach())
            all_targs.append(targets.detach())

            del logits, batch
            torch.cuda.empty_cache()

    # Concatenate local results
    if all_preds:
        local_preds = torch.cat(all_preds, dim=0)
        local_targs = torch.cat(all_targs, dim=0)
    else:
        # Edge case: this rank had no data
        local_preds = torch.empty(0, device=accelerator.device)
        local_targs = torch.empty(0, device=accelerator.device)

    # Pad local tensors to uniform size for gather
    world_size = accelerator.num_processes
    per_rank = math.ceil(actual_n_samples / world_size)
    pad_size = per_rank - local_preds.shape[0]
    if pad_size > 0:
        pad_shape = (pad_size,) + local_preds.shape[1:]
        local_preds = torch.cat([
            local_preds,
            torch.zeros(pad_shape, device=local_preds.device,
                        dtype=local_preds.dtype)
        ], dim=0)
        local_targs = torch.cat([
            local_targs,
            torch.zeros(pad_shape, device=local_targs.device,
                        dtype=local_targs.dtype)
        ], dim=0)

    # Gather across all ranks
    gathered_preds = accelerator.gather(local_preds)
    gathered_targs = accelerator.gather(local_targs)

    if is_main:
        # Trim padding and convert to numpy
        predictions = gathered_preds[:actual_n_samples].cpu().numpy()
        targets = gathered_targs[:actual_n_samples].cpu().numpy()

        # Summary diagnostics
        n_total = predictions.size
        n_pred_nan = int(np.isnan(predictions).sum())
        if nan_batches > 0:
            print(f"    WARNING: {nan_batches}/{i + 1} batches "
                  f"had NaN logits")
        if n_pred_nan > 0:
            print(f"    WARNING: {n_pred_nan}/{n_total} NaN values "
                  f"in final predictions array")
        else:
            print(f"    predictions: min={predictions.min():.4f}, "
                  f"max={predictions.max():.4f}, "
                  f"mean={predictions.mean():.4f}")

        return predictions, targets

    return None, None


def save_predictions_csv(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    task: str,
    phenotype_names: Optional[List[str]] = None
):
    """Save predictions and targets side-by-side in a CSV file.

    Args:
        predictions: Array of shape (n_samples,) or (n_samples, n_classes).
        targets: Array with the same shape as predictions.
        output_path: File path for the output CSV.
        task: Task name, used for deriving column names.
        phenotype_names: Optional list of phenotype class names used as
            column suffixes for the 'phenotype' task.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    n_classes = predictions.shape[1]

    if task == 'phenotype' and phenotype_names is not None \
            and len(phenotype_names) == n_classes:
        pred_cols = [f'pred_{name}' for name in phenotype_names]
        targ_cols = [f'target_{name}' for name in phenotype_names]
    elif n_classes > 1:
        pred_cols = [f'prediction_{i}' for i in range(n_classes)]
        targ_cols = [f'target_{i}' for i in range(n_classes)]
    else:
        pred_cols = ['prediction']
        targ_cols = ['target']

    df = pd.DataFrame(
        np.hstack([predictions, targets]),
        columns=pred_cols + targ_cols
    )
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extract predictions from finetuned TransEHR2 models'
    )
    parser.add_argument(
        'dataset_config', type=str,
        help='YAML file specifying dataset parameters'
    )
    parser.add_argument(
        'experiment_config', type=str,
        help='YAML file specifying experiment/model architecture parameters'
    )
    parser.add_argument(
        'experiment_name', type=str,
        help='Name of the experiment (locates model weights under model_dir)'
    )
    parser.add_argument(
        '--model_dir', type=str, default='./models',
        help='Root directory containing saved model weights '
             '(default: ./models)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Number of DataLoader worker processes (default: 0)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Batch size for inference. If not specified, the value from '
             'the experiment config is used.'
    )
    args = parser.parse_args()

    # ---- Load configuration files ----
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    with open(args.experiment_config, 'r') as f:
        experiment_config = yaml.safe_load(f)

    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    TEXT_FEATS = dataset_config['TEXT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']

    USE_TEXT = experiment_config['USE_TEXT']
    BATCH_SIZE = args.batch_size or experiment_config['BATCH_SIZE']
    MODEL_DIR = args.model_dir
    EXPERIMENT_NAME = args.experiment_name

    # ---- Compute feature dimensions ----
    with open(VARIABLE_PROPERTIES_PATH, 'r') as f:
        variable_properties = yaml.safe_load(f)

    tot_val_feat_dim = 0
    for feature in VALUED_FEATS:
        tot_val_feat_dim += variable_properties[feature]['size']
    if USE_TEXT:
        n_val_feats = len(VALUED_FEATS) + len(TEXT_FEATS)
        tot_val_feat_dim += len(TEXT_FEATS) * TEXT_EMBED_DIM
    else:
        n_val_feats = len(VALUED_FEATS)
    n_event_types = len(EVENT_FEATS)

    # ---- Initialize Accelerator ----
    accelerator = initialize_inference_accelerator()

    if accelerator.is_main_process:
        print(f"Device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Model directory: {MODEL_DIR}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Batch size: {BATCH_SIZE}\n")

    # ---- Iterate over folds ----
    fold_names = get_fold_names(DATA_DIR, exclude=['fold0'])
    if not fold_names:
        if accelerator.is_main_process:
            print(f"No fold directories found in {DATA_DIR}")
        exit(1)

    for fold_name in fold_names:
        # Recreate accelerator between folds to free resources
        if 'accelerator' in locals() and accelerator is not None:
            accelerator.free_memory()
            del accelerator
            gc.collect()
            torch.cuda.empty_cache()
        accelerator = initialize_inference_accelerator()

        if accelerator.is_main_process:
            print(f"{'=' * 60}")
            print(f"Fold: {fold_name}")
            print(f"{'=' * 60}")

        fold_dir = os.path.join(DATA_DIR, fold_name)

        # Load DataLoaders for each available split (sharded by rank)
        pin_memory = accelerator.device.type == 'cuda'
        loaders: Dict[str, DataLoader] = {}
        dataset_sizes: Dict[str, int] = {}
        for split in ['train', 'val', 'test']:
            loader, total = create_inference_loader(
                fold_dir, split, BATCH_SIZE,
                args.num_workers, pin_memory,
                world_size=accelerator.num_processes,
                rank=accelerator.process_index
            )
            if loader is not None:
                loaders[split] = loader
                dataset_sizes[split] = total

        # Read phenotype class names for CSV column headers
        phenotype_names = get_phenotype_names(fold_dir)

        # Determine number of phenotype output classes from the dataset
        any_loader = next(iter(loaders.values()))
        # Subset wraps the original dataset; unwrap to access attributes
        base_dataset = any_loader.dataset
        if isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        phenotype_arr_shape = base_dataset.phenotype.shape
        n_phenotype_classes = (phenotype_arr_shape[1]
                              if len(phenotype_arr_shape) > 1 else 1)

        # ---- Iterate over prediction tasks ----
        for task in ['mortality', 'length_of_stay', 'phenotype']:
            if accelerator.is_main_process:
                print(f"\n  Task: {task}")

            num_classes = (n_phenotype_classes
                          if task == 'phenotype' else 1)

            # Build a fresh model with random weights
            model = build_classifier(
                experiment_config,
                n_val_feats=n_val_feats,
                tot_val_feat_dim=tot_val_feat_dim,
                n_event_types=n_event_types,
                n_static_feats=len(STATIC_FEATS),
                num_classes=num_classes,
                use_text=USE_TEXT
            )

            # Load finetuned weights
            weights_path = os.path.join(
                MODEL_DIR, EXPERIMENT_NAME, fold_name,
                'pretrained', f'finetuned_{task}.pt'
            )
            if not load_finetuned_weights(
                model, weights_path, USE_TEXT
            ):
                if accelerator.is_main_process:
                    print(f"    WARNING: Weights not found at "
                          f"{weights_path}, skipping.")
                del model
                gc.collect()
                continue

            if accelerator.is_main_process:
                print(f"    Loaded weights from {weights_path}")

            # Wrap model with Accelerate for distributed inference
            model = accelerator.prepare(model)

            # Install NaN hooks on the unwrapped model so they fire
            # correctly through the DDP/FSDP wrapper
            unwrapped = accelerator.unwrap_model(model)
            hooks = install_nan_hooks(unwrapped)
            nan_counts = hooks[0].nan_counts  # shared counter dict

            accelerator.wait_for_everyone()

            # Run inference on each data split
            for split, loader in loaders.items():
                actual_n = dataset_sizes[split]
                if accelerator.is_main_process:
                    print(f"    Split: {split} ({actual_n} samples)")

                predictions, targets = run_inference(
                    model, loader, task, accelerator, actual_n
                )

                if accelerator.is_main_process:
                    output_path = os.path.join(
                        MODEL_DIR, EXPERIMENT_NAME, fold_name,
                        task,
                        f'{task}_{split}_finetuned_output.csv'
                    )
                    save_predictions_csv(
                        predictions, targets, output_path,
                        task, phenotype_names
                    )
                    print(f"    -> {output_path}")

                accelerator.wait_for_everyone()

            # Report hook activity and clean up
            if accelerator.is_main_process:
                for enc_name, cnt in nan_counts.items():
                    if cnt > 0:
                        print(f"    NaN->0 replacements in "
                              f"{enc_name}: {cnt}")
            for h in hooks:
                h.remove()

            # Free model memory before next task
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free dataloader memory before next fold
        del loaders
        gc.collect()

    if accelerator.is_main_process:
        print(f"\n{'=' * 60}")
        print("Done.")
