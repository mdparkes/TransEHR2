"""
Extract predictions from finetuned TransEHR2 models for evaluation.

For each cross-validation fold and prediction task (mortality, length_of_stay,
phenotype), loads the finetuned model weights, performs a forward pass on
training, validation (if available), and test data, and saves predictions
and targets to CSV files.

Output files are written to:
    {model_dir}/{experiment_name}/{fold}/{task}/{task}_{split}_finetuned_output.csv

Usage:
    python dump_finetuned_predictions.py <dataset_config> <experiment_config> <experiment_name> \
        [--model_dir ./models] [--num_workers 0] [--batch_size 750]
"""

import argparse
import gc
import numpy as np
import os
import pandas as pd
import re
import torch
import yaml

from collections import OrderedDict
from torch import Tensor
from torch.utils.data import DataLoader
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


def get_device() -> torch.device:
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_dataloader(
    fold_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> Optional[DataLoader]:
    """Create a DataLoader for one data split within a fold.

    All DataLoaders use sequential (non-shuffled) ordering so that
    predictions can be deterministically aligned with targets.

    Args:
        fold_dir: Path to the fold directory containing split subdirectories.
        split: One of 'train', 'val', or 'test'.
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to use pinned memory for CUDA transfers.

    Returns:
        A DataLoader, or None if the split directory does not exist (only
        allowed for 'val').

    Raises:
        FileNotFoundError: If a required split directory ('train' or 'test')
            is not found.
    """
    dataset_path = os.path.join(fold_dir, split)
    if not os.path.exists(dataset_path):
        if split == 'val':
            return None
        raise FileNotFoundError(f"'{split}/' not found in {fold_dir}")

    dataset = load_dataset(dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tensorized,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )


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
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and collect predictions and targets.

    For binary classification tasks (mortality, phenotype), predictions are
    sigmoid-transformed probabilities. For regression (length_of_stay),
    predictions are raw model outputs.

    Args:
        model: The finetuned MixedClassifier in eval mode.
        loader: DataLoader for the data split.
        task: One of 'mortality', 'length_of_stay', or 'phenotype'.
        device: Compute device.

    Returns:
        Tuple of (predictions, targets) as numpy arrays with shape
        (n_samples, n_outputs).
    """
    model.eval()
    all_preds = []
    all_targs = []
    nan_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(loader, desc=f'    Inference', leave=False)
        ):
            batch = move_batch_to_device(batch, device=device)
            logits = model(batch)
            targets = batch['targets'][task]

            if task in ('mortality', 'phenotype'):
                preds = torch.sigmoid(logits)
            else:
                preds = logits

            n_nan = torch.isnan(logits).sum().item()
            if n_nan > 0:
                nan_batches += 1

            # Diagnostics on the first batch
            if i == 0:
                if n_nan > 0:
                    print(f"    WARNING: {n_nan} NaN values in logits "
                          f"(batch 0, shape {tuple(logits.shape)})")
                    print(f"    logits sample: {logits[0].cpu().tolist()}")
                else:
                    print(f"    logits OK (batch 0): "
                          f"min={logits.min().item():.4f}, "
                          f"max={logits.max().item():.4f}")

            all_preds.append(preds.cpu().numpy())
            all_targs.append(targets.cpu().numpy())

            del logits, batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targs, axis=0)

    # Summary diagnostics
    n_total = predictions.size
    n_pred_nan = int(np.isnan(predictions).sum())
    if nan_batches > 0:
        print(f"    WARNING: {nan_batches}/{i + 1} batches had NaN logits")
    if n_pred_nan > 0:
        print(f"    WARNING: {n_pred_nan}/{n_total} NaN values in "
              f"final predictions array")
    else:
        print(f"    predictions: min={predictions.min():.4f}, "
              f"max={predictions.max():.4f}, "
              f"mean={predictions.mean():.4f}")

    return predictions, targets


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

    # ---- Device selection ----
    device = get_device()
    print(f"Device: {device}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Batch size: {BATCH_SIZE}\n")

    # ---- Iterate over folds ----
    fold_names = get_fold_names(DATA_DIR, exclude=['fold0'])
    if not fold_names:
        print(f"No fold directories found in {DATA_DIR}")
        exit(1)

    for fold_name in fold_names:
        print(f"{'=' * 60}")
        print(f"Fold: {fold_name}")
        print(f"{'=' * 60}")

        fold_dir = os.path.join(DATA_DIR, fold_name)

        # Load DataLoaders for each available split (no shuffling)
        pin_memory = device.type == 'cuda'
        loaders: Dict[str, DataLoader] = {}
        for split in ['train', 'val', 'test']:
            loader = create_dataloader(
                fold_dir, split, BATCH_SIZE,
                args.num_workers, pin_memory
            )
            if loader is not None:
                loaders[split] = loader

        # Read phenotype class names for CSV column headers
        phenotype_names = get_phenotype_names(fold_dir)

        # Determine number of phenotype output classes from the dataset
        any_dataset = next(iter(loaders.values())).dataset
        phenotype_arr_shape = any_dataset.phenotype.shape
        n_phenotype_classes = (phenotype_arr_shape[1]
                              if len(phenotype_arr_shape) > 1 else 1)

        # ---- Iterate over prediction tasks ----
        for task in ['mortality', 'length_of_stay', 'phenotype']:
            print(f"\n  Task: {task}")

            num_classes = n_phenotype_classes if task == 'phenotype' else 1

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
            if not load_finetuned_weights(model, weights_path, USE_TEXT):
                print(f"    WARNING: Weights not found at {weights_path}, "
                      f"skipping.")
                del model
                gc.collect()
                continue

            model = model.to(device)
            print(f"    Loaded weights from {weights_path}")

            # Install forward hooks that replace NaN encoder output with
            # zeros.  This prevents NaN from propagating through the
            # padding-mask multiplication and aggregation sum when all
            # episodes in a batch have padding at a given timestep.
            hooks = install_nan_hooks(model)
            nan_counts = hooks[0].nan_counts  # shared counter dict

            # Run inference on each data split
            for split, loader in loaders.items():
                n_samples = len(loader.dataset)
                print(f"    Split: {split} ({n_samples} samples)")

                predictions, targets = run_inference(
                    model, loader, task, device
                )

                output_path = os.path.join(
                    MODEL_DIR, EXPERIMENT_NAME, fold_name, task,
                    f'{task}_{split}_finetuned_output.csv'
                )
                save_predictions_csv(
                    predictions, targets, output_path,
                    task, phenotype_names
                )
                print(f"    -> {output_path}")

            # Report hook activity and clean up
            for enc_name, cnt in nan_counts.items():
                if cnt > 0:
                    print(f"    NaN→0 replacements in {enc_name}: {cnt}")
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

    print(f"\n{'=' * 60}")
    print("Done.")
