"""
Accelerate-compatible training routines for distributed training.
"""

import accelerate.utils
import json
import numpy as np
import os
import shutil
import torch

from accelerate import Accelerator
from accelerate.utils import broadcast, broadcast_object_list
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import Tensor
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from typing import Any, Dict, Optional, OrderedDict, Tuple, TypeAlias

from TransEHR2.losses import MaskedDiscriminatorLoss, MaskedGeneratorLoss, TransformerHawkesLoss
from TransEHR2.models import MixedClassifier
from TransEHR2.utils import DistributedTimer
from TransEHR2.utils import format_pretraining_performance_table, generate_record_masks, get_param_shapes
from TransEHR2.utils import print_peak_memory


MetadataDict: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = OrderedDict[str, Tensor]


def is_fsdp_enabled(accelerator: Accelerator) -> bool:
    """Check if the accelerator is using FSDP.
    
    Args:
        accelerator: The Accelerate Accelerator object.
    
    Returns:
        True if FSDP is enabled, False otherwise (e.g., DDP/MULTI_GPU mode).
    """
    return accelerator.distributed_type == accelerate.utils.DistributedType.FSDP


def reshape_flattened_state_dict(
    state_dict: StateDict,
    param_shapes: OrderedDict[str, tuple]
) -> StateDict:
    """Reshape flattened FSDP state dict to match expected parameter shapes.
    
    NOTE: LLM parameters are intentionally excluded from state dicts since
    the LLM is frozen and always initialized from HuggingFace weights.
    
    This function is primarily needed for FSDP, which may flatten parameters.
    For DDP, parameters retain their original shapes.
    """
    
    def strip_fsdp_prefix(key: str) -> str:
        """Remove FSDP wrapper prefixes from parameter names."""
        # Remove standard FSDP wrapper prefixes
        for prefix in ['_fsdp_wrapped_module.', '_forward_module.', 'module.']:
            if prefix in key:
                key = key.replace(prefix, '')
        
        # Fix FSDP's flattening of llm_module.model -> llm_model
        # FSDP sometimes flattens the hierarchy: llm_module (GradientTraceableLLM) -> model (AutoModel)
        # becomes just llm_model instead of llm_module.model
        if key.startswith('llm_model.'):
            key = key.replace('llm_model.', 'llm_module.model.')
        
        return key
    
    reshaped_state_dict = OrderedDict()
    
    for key, tensor in state_dict.items():
        clean_key = strip_fsdp_prefix(key)
        
        # Ensure tensor is on CPU
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        
        # Check if we have an expected shape for this key
        if clean_key in param_shapes:
            expected_shape = param_shapes[clean_key]
            
            # Reshape if shapes don't match
            if tensor.shape != expected_shape:
                # Verify total elements match before reshaping
                if tensor.numel() != torch.prod(torch.tensor(expected_shape)).item():
                    print(f"ERROR: Cannot reshape {clean_key}: tensor has {tensor.numel()} elements but expected shape {expected_shape} has {torch.prod(torch.tensor(expected_shape)).item()} elements")
                    # Keep original shape if reshape would fail
                    reshaped_state_dict[clean_key] = tensor.clone()
                else:
                    print(f"Reshaping {clean_key}: {tensor.shape} -> {expected_shape}")
                    reshaped_state_dict[clean_key] = tensor.reshape(expected_shape).clone()
            else:
                reshaped_state_dict[clean_key] = tensor.clone()
        else:
            # Only warn if it's not an LLM parameter
            if not (clean_key.startswith('llm_module.') or clean_key.startswith('llm_model.')):
                print(f"Warning: No expected shape for {clean_key}, keeping original shape {tensor.shape}")
            reshaped_state_dict[clean_key] = tensor.clone()
    
    return reshaped_state_dict


def extract_state_dict(
    accelerator: Accelerator,
    model: torch.nn.Module,
    param_shapes: OrderedDict[str, tuple]
) -> Optional[StateDict]:
    """Extract state dict from model, handling both FSDP and DDP modes.
    
    For FSDP: Uses state_dict_type context manager for proper gathering.
    For DDP: Directly accesses state dict on main process.
    
    Args:
        accelerator: The Accelerate Accelerator object.
        model: The wrapped model (may be FSDP or DDP wrapped).
        param_shapes: Expected parameter shapes for reshaping (FSDP only).
    
    Returns:
        State dict on main process (with LLM params filtered out), None on other ranks.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    
    if is_fsdp_enabled(accelerator):
        # FSDP: use state_dict_type context manager for proper gathering
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, cfg):
            # All ranks call this (all-gather op), but only rank 0 receives result
            state_dict = unwrapped_model.state_dict()
            if accelerator.is_main_process:
                state_dict_no_llm = OrderedDict()
                for key, value in state_dict.items():
                    if not key.startswith('llm_module.'):
                        state_dict_no_llm[key] = value
                return reshape_flattened_state_dict(state_dict_no_llm, param_shapes)
            else:
                return None
    else:
        # DDP: state dict is already complete on each rank, just grab it on main
        if accelerator.is_main_process:
            state_dict = unwrapped_model.state_dict()
            state_dict_no_llm = OrderedDict()
            for key, value in state_dict.items():
                if not key.startswith('llm_module.'):
                    state_dict_no_llm[key] = value.cpu().clone()
            return state_dict_no_llm  # No reshape needed for DDP
        else:
            return None


def load_model_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: str,
    metadata_path: str,
    best_state_path: Optional[str] = None,
    timer: Optional[DistributedTimer] = None
) -> Tuple[MetadataDict, Optional[StateDict]]:
    """Load checkpoint data from a model (pretraining or finetuning).

    This function uses the `Accelerator` object to load sharded model, optimizer, and scheduler 
    states. It also loads training metadata (such as epoch number and best validation losses/metrics) 
    from a JSON file, broadcasting the metadata to all processes to ensure synchronization. If a path 
    to a state dictionary of parameters from the best-performing epoch is provided, the main process 
    loads the state dictionary, mapping it to CPU. It is mapped to the CPU to lower the memory usage 
    on GPUs, and the assumption is that the state dict is not actually loaded into the model inside 
    any of the routines functions, but rather saved for use in the scripts that call these routines. 
    If no metadata file is found, an empty dictionary is returned on all ranks.

    Args:
        accelerator: The Accelerate `Accelerator` object managing distributed training.
        checkpoint_path: Path to the checkpoint directory containing model, optimizer, and scheduler states.
        metadata_path: Path to the JSON file containing training metadata.
        best_state_path: Path to the state dictionary of the best-performing epoch. Defaults to None.
        timer: Timer for tracking execution time. Defaults to None.
    
    Returns:
        Tuple of (metadata_dict, best_state_dict) where both are synchronized across all ranks
    """

    # Check whether files exist - avoids file access timing issues with different ranks
    if accelerator.is_main_process:
        metadata_exists = os.path.exists(metadata_path)
        checkpoint_exists = os.path.exists(checkpoint_path)
        best_state_dict_exists = best_state_path is not None and os.path.exists(best_state_path)
    else:
        metadata_exists = False
        checkpoint_exists = False
        best_state_dict_exists = False
    # Broadcast file checks
    checks_list = broadcast_object_list([metadata_exists, checkpoint_exists, best_state_dict_exists], from_process=0)
    metadata_exists = checks_list[0]
    checkpoint_exists = checks_list[1]
    best_state_dict_exists = checks_list[2]

    if not (checkpoint_exists and metadata_exists):
        if accelerator.is_main_process:
            print("\nOne of checkpoint or training metadata not found. Training from scratch.")
        return {}, None

    # Load training metadata
    if accelerator.is_main_process:
        with open(metadata_path, 'r') as f_in:
            metadata = json.load(f_in)

        loaded_data = {
            'start_epoch': metadata.get('start_epoch', metadata.get('epoch', 0) + 1),
            'best_epoch': metadata.get('best_epoch', -1),
            'early_stopping_counter': metadata.get('early_stopping_counter', 0)
        }

        # Add pretraining-specific fields if present
        if 'best_epoch_train_losses' in metadata:
            loaded_data['best_epoch_train_losses'] = metadata['best_epoch_train_losses']
            loaded_data['best_epoch_val_losses'] = metadata['best_epoch_val_losses']

        # Add finetuning-specific fields if present
        if 'best_val_metric' in metadata:
            loaded_data['best_epoch_val_metric'] = metadata['best_val_metric']
        if 'best_train_scores' in metadata:
            loaded_data['best_epoch_train_scores'] = metadata['best_train_scores']
            loaded_data['best_epoch_val_scores'] = metadata['best_val_scores']
        if 'task' in metadata:
            loaded_data['task'] = metadata['task']
        
        # Update timer if provided
        if timer is not None:
            timer.restore_from_checkpoint(metadata, is_main_process=accelerator.is_main_process)
    else:
        loaded_data = None
    
    # Broadcast loaded metadata to all ranks
    loaded_data = broadcast_object_list([loaded_data], from_process=0)[0]
    
    # Load model/optimizer/scheduler state only if checkpoint exists
    accelerator.load_state(checkpoint_path)
    if accelerator.is_main_process:
        print(f"\nResuming from checkpoint at epoch {loaded_data['start_epoch']}")
    
    # Load unsharded best state dict on CPU (main process only)
    best_state_dict = None
    if best_state_dict_exists:
        if accelerator.is_main_process:
            # Whitelist OrderedDict as safe for loading
            with torch.serialization.safe_globals([OrderedDict]):
                best_state_dict = torch.load(best_state_path, map_location='cpu')
                print(f"Loaded best state dict on CPU from epoch {loaded_data['best_epoch']}\n")

    accelerator.wait_for_everyone()
    
    return loaded_data, best_state_dict


def save_model_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: str,
    metadata_path: str,
    epoch: int,
    early_stopping_counter: int = 0,
    best_epoch: Optional[int] = None,
    best_state_dict: Optional[StateDict] = None,
    best_state_path: Optional[str] = None,
    timer: Optional[DistributedTimer] = None,
    # Pretraining-specific (optional)
    best_epoch_train_losses: Optional[Dict[str, float]] = None,
    best_epoch_val_losses: Optional[Dict[str, float]] = None,
    # Finetuning-specific (optional)
    best_epoch_val_metric: Optional[float] = None,
    best_epoch_train_scores: Optional[Dict[str, Any]] = None,
    best_epoch_val_scores: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None
) -> None:
    """Save checkpoint data for a model (pretraining or finetuning).

    This function uses the `Accelerator` object to save sharded model, optimizer, and scheduler 
    states. It also saves training metadata (such as epoch number and best validation losses/metrics) 
    to a JSON file. The main process also optionally saves the state dictionary of the best-performing 
    epoch if provided, mapping it to CPU to lower GPU memory usage.

    Args:
        accelerator: The Accelerate `Accelerator` object managing distributed training.
        checkpoint_path: Path to save the checkpoint directory containing model, optimizer, and scheduler states.
        metadata_path: Path to save the JSON file containing training metadata.
        epoch: Current epoch number. When resuming from a checkpoint, the start epoch will be `epoch + 1`.
        best_epoch: Epoch number of the best-performing model.
        early_stopping_counter: Current early stopping counter. Defaults to 0.
        best_state_dict: State dictionary of the best-performing epoch. Defaults to None.
        best_state_path: Path to save the state dictionary of the best-performing epoch. 
            Defaults to None. Must be provided if `best_state_dict` is provided.
        timer: Optional timer object for tracking distributed training time. Defaults to None.
        best_epoch_train_losses: Training losses from best epoch (pretraining). Defaults to None.
        best_epoch_val_losses: Validation losses from best epoch (pretraining). Defaults to None.
        best_epoch_val_metric: Best validation metric (finetuning). Defaults to None.
        best_epoch_train_scores: Training scores from best epoch (finetuning). Defaults to None.
        best_epoch_val_scores: Validation scores from best epoch (finetuning). Defaults to None.
        task: Task name for finetuning ('mortality', 'length_of_stay', or 'phenotype'). Defaults to None.
    
    Returns:
        None
    """
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy/torch types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert numpy scalar to Python scalar
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        else:
            return obj
    
    # Save checkpoint using Accelerator -- all ranks participate
    accelerator.save_state(checkpoint_path)
    
    # Main process saves metadata and best state dict (if provided)
    if accelerator.is_main_process:
        # Prepare MetadataDict with common fields
        metadata = {
            'start_epoch': epoch + 1,
            'early_stopping_counter': early_stopping_counter
        }
        if best_epoch is not None:
            metadata['best_epoch'] = best_epoch
        
        # Add pretraining-specific fields if provided
        if best_epoch_train_losses is not None:
            metadata['best_epoch_train_losses'] = convert_to_json_serializable(best_epoch_train_losses)
        if best_epoch_val_losses is not None:
            metadata['best_epoch_val_losses'] = convert_to_json_serializable(best_epoch_val_losses)
            
        # Add finetuning-specific fields if provided
        if best_epoch_val_metric is not None:
            metadata['best_epoch_val_metric'] = convert_to_json_serializable(best_epoch_val_metric)
        if best_epoch_train_scores is not None:
            metadata['best_epoch_train_scores'] = convert_to_json_serializable(best_epoch_train_scores)
        if best_epoch_val_scores is not None:
            metadata['best_epoch_val_scores'] = convert_to_json_serializable(best_epoch_val_scores)
        if task is not None:
            metadata['task'] = task
        
        # Store timer state if provided
        if timer is not None:
            timer_state = timer.get_timer_state_for_checkpoint()
            metadata.update(convert_to_json_serializable(timer_state))
        
        # Write metadata
        with open(metadata_path, 'w') as f_out:
            json.dump(metadata, f_out, indent=2)
        
        # Write best_state_dict if provided
        if best_state_dict is not None and best_state_path is not None:
            torch.save(best_state_dict, best_state_path)
        
        print(f"\nSaved checkpoint at epoch {epoch + 1}")
    
    accelerator.wait_for_everyone()


def save_encoder_weights(
    accelerator: Accelerator,
    best_state_dict: Optional[StateDict],
    save_dir: str
) -> None:
    """Save encoder weights from a reshaped state dict.
    
    Args:
        accelerator: The Accelerate Accelerator object.
        best_state_dict: Full reshaped state dict (already on CPU). None on non-main ranks.
        save_dir: Directory to save encoder weights.
    """
    
    # Only main process does extraction and saving
    if accelerator.is_main_process:
        if best_state_dict is None:
            print("Warning: best_state_dict is None")
            accelerator.wait_for_everyone()
            return
        
        value_encoder_state = {}
        event_encoder_state = {}
        
        # Extract value encoder params (already reshaped)
        for key, value in best_state_dict.items():
            if key.startswith('discriminator.encoder.'):
                local_name = key.replace('discriminator.encoder.', '')
                value_encoder_state[local_name] = value.cpu().clone()
            elif key.startswith('hawkes.encoder.'):
                local_name = key.replace('hawkes.encoder.', '')
                event_encoder_state[local_name] = value.cpu().clone()
        
        if len(value_encoder_state) == 0 or len(event_encoder_state) == 0:
            print(f"Warning: No encoder weights found")
            accelerator.wait_for_everyone()
            return
        
        # Save
        torch.save(value_encoder_state, os.path.join(save_dir, 'value_encoder.pt'))
        torch.save(event_encoder_state, os.path.join(save_dir, 'event_encoder.pt'))
        
        print(f"\nSaved encoder weights to {save_dir}")
    
    accelerator.wait_for_everyone()


def pretrain_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    gen_loss_fn: MaskedGeneratorLoss,
    disc_loss_fn: MaskedDiscriminatorLoss,
    thp_loss_fn: TransformerHawkesLoss,
    thp_loss_mc_samples: int,
    record_mask_ratio: float,
    obs_unobs_sample_ratio: float,
    cmpnt_mask_ratio: float,
    accelerator: Accelerator,
    desc: str = "Training",
    mem_test_mode: bool = False
) -> Dict[str, float]:
    """Execute one epoch of pretraining.
    
    Returns dictionary of average losses for the epoch.
    """
    model.train()
    
    train_losses = []
    train_gen_losses = []
    train_disc_losses = []
    train_thp_losses = []
    train_thp_nll_losses = []
    train_thp_type_losses = []
    train_thp_time_losses = []

    disable_tqdm = not accelerator.is_local_main_process

    for batch in tqdm(loader, desc=desc, leave=False, disable=disable_tqdm):
        
        value_associated_data_masks, _ = generate_record_masks(
            batch,
            feature_sample_rate=record_mask_ratio,
            obs_unobs_ratio=obs_unobs_sample_ratio,
            subsample_rate=cmpnt_mask_ratio
        )
        
        electra_output = model(
            batch,
            value_associated_data_masks,
            device=accelerator.device,
            trace_grads=False,
            compute_intensities=True,
            thp_loss_mc_samples=thp_loss_mc_samples
        )
        
        generator_preds = electra_output['generator']
        discriminator_preds = electra_output['discriminator']
        thp_encodings = electra_output.get('hawkes_encodings', None)
        thp_preds = electra_output.get('hawkes_predictions', (None, None))
        
        if thp_preds != (None, None):
            thp_type_preds, thp_time_preds = thp_preds
        else:
            thp_type_preds, thp_time_preds = (None, None)
        
        gen_loss = gen_loss_fn(generator_preds, batch, value_associated_data_masks)
        disc_loss = disc_loss_fn(discriminator_preds, value_associated_data_masks)
        
        if thp_encodings is not None and 'thp_intensities' in electra_output:
            intensities = electra_output['thp_intensities']
            event_data = batch['event_data']

            thp_loss, (thp_nll_loss, thp_type_loss, thp_time_loss) = thp_loss_fn(
                intensities['obs_initial'],
                intensities['obs_conditional'],
                intensities['sampled'],
                event_data,
                thp_type_preds,
                thp_time_preds
            )
            loss = gen_loss + disc_loss + thp_loss
            train_thp_losses.append(thp_loss.item())
            train_thp_nll_losses.append(thp_nll_loss.item())
            train_thp_type_losses.append(thp_type_loss.item())
            train_thp_time_losses.append(thp_time_loss.item())
        else:
            loss = gen_loss + disc_loss
            train_thp_losses.append(0.0)
            train_thp_nll_losses.append(0.0)
            train_thp_type_losses.append(0.0)
            train_thp_time_losses.append(0.0)

        train_losses.append(loss.item())
        train_gen_losses.append(gen_loss.item())
        train_disc_losses.append(disc_loss.item())

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if mem_test_mode:
            print_peak_memory(accelerator)
            raise Exception("Memory test mode enabled, terminating after one batch.")

    # Gather and average losses across all ranks
    train_losses = accelerator.gather(torch.tensor(train_losses, device=accelerator.device))
    train_gen_losses = accelerator.gather(torch.tensor(train_gen_losses, device=accelerator.device))
    train_disc_losses = accelerator.gather(torch.tensor(train_disc_losses, device=accelerator.device))
    train_thp_losses = accelerator.gather(torch.tensor(train_thp_losses, device=accelerator.device))
    train_thp_nll_losses = accelerator.gather(torch.tensor(train_thp_nll_losses, device=accelerator.device))
    train_thp_type_losses = accelerator.gather(torch.tensor(train_thp_type_losses, device=accelerator.device))
    train_thp_time_losses = accelerator.gather(torch.tensor(train_thp_time_losses, device=accelerator.device))
    
    # Compute mean on main process, keep as tensor for broadcast
    if accelerator.is_main_process:
        curr_train_loss = train_losses.mean()
        curr_train_gen_loss = train_gen_losses.mean()
        curr_train_disc_loss = train_disc_losses.mean()
        curr_train_thp_loss = train_thp_losses.mean()
        curr_train_thp_nll_loss = train_thp_nll_losses.mean()
        curr_train_thp_type_loss = train_thp_type_losses.mean()
        curr_train_thp_time_loss = train_thp_time_losses.mean()
    else:
        # Create dummy tensors on other ranks
        curr_train_loss = torch.tensor(0.0, device=accelerator.device)
        curr_train_gen_loss = torch.tensor(0.0, device=accelerator.device)
        curr_train_disc_loss = torch.tensor(0.0, device=accelerator.device)
        curr_train_thp_loss = torch.tensor(0.0, device=accelerator.device)
        curr_train_thp_nll_loss = torch.tensor(0.0, device=accelerator.device)
        curr_train_thp_type_loss = torch.tensor(0.0, device=accelerator.device)
        curr_train_thp_time_loss = torch.tensor(0.0, device=accelerator.device)
    
    # Broadcast tensors from main process to all ranks, then convert to float
    curr_train_loss = broadcast(curr_train_loss, from_process=0).item()
    curr_train_gen_loss = broadcast(curr_train_gen_loss, from_process=0).item()
    curr_train_disc_loss = broadcast(curr_train_disc_loss, from_process=0).item()
    curr_train_thp_loss = broadcast(curr_train_thp_loss, from_process=0).item()
    curr_train_thp_nll_loss = broadcast(curr_train_thp_nll_loss, from_process=0).item()
    curr_train_thp_type_loss = broadcast(curr_train_thp_type_loss, from_process=0).item()
    curr_train_thp_time_loss = broadcast(curr_train_thp_time_loss, from_process=0).item()
    
    return {
        'Optimization_Loss': curr_train_loss,
        'Generator_Loss': curr_train_gen_loss,
        'Discriminator_Loss': curr_train_disc_loss,
        'THP_Loss': curr_train_thp_loss,
        'THP_NLL_Loss': curr_train_thp_nll_loss,
        'THP_Type_Loss': curr_train_thp_type_loss,
        'THP_Time_Loss': curr_train_thp_time_loss
    }


def pretrain_validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    gen_loss_fn: MaskedGeneratorLoss,
    disc_loss_fn: MaskedDiscriminatorLoss,
    thp_loss_fn: TransformerHawkesLoss,
    thp_loss_mc_samples: int,
    record_mask_ratio: float,
    obs_unobs_sample_ratio: float,
    cmpnt_mask_ratio: float,
    accelerator: Accelerator,
    desc: str = "Validation"
) -> Dict[str, float]:
    """Execute one epoch of pretraining validation.
    
    Returns dictionary of average validation losses.
    """

    model.eval()
    
    val_losses = []
    val_gen_losses = []
    val_disc_losses = []
    val_thp_losses = []
    val_thp_nll_losses = []
    val_thp_type_losses = []
    val_thp_time_losses = []

    disable_tqdm = not accelerator.is_local_main_process

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, disable=disable_tqdm):

            value_associated_data_masks, _ = generate_record_masks(
                batch,
                feature_sample_rate=record_mask_ratio,
                obs_unobs_ratio=obs_unobs_sample_ratio,
                subsample_rate=cmpnt_mask_ratio
            )
            
            electra_output = model(
                batch,
                value_associated_data_masks,
                device=accelerator.device,
                trace_grads=False,
                compute_intensities=True,
                thp_loss_mc_samples=thp_loss_mc_samples
            )
            
            generator_preds = electra_output['generator']
            discriminator_preds = electra_output['discriminator']
            thp_encodings = electra_output.get('hawkes_encodings', None)
            thp_preds = electra_output.get('hawkes_predictions', (None, None))
            
            if thp_preds != (None, None):
                thp_type_preds, thp_time_preds = thp_preds
            else:
                thp_type_preds, thp_time_preds = (None, None)
            
            gen_loss = gen_loss_fn(generator_preds, batch, value_associated_data_masks)
            disc_loss = disc_loss_fn(discriminator_preds, value_associated_data_masks)
            
            if thp_encodings is not None and 'thp_intensities' in electra_output:
                intensities = electra_output['thp_intensities']
                event_data = batch['event_data']

                thp_loss, (thp_nll_loss, thp_type_loss, thp_time_loss) = thp_loss_fn(
                    intensities['obs_initial'],
                    intensities['obs_conditional'],
                    intensities['sampled'],
                    event_data,
                    thp_type_preds,
                    thp_time_preds
                )
                loss = gen_loss + disc_loss + thp_loss
                val_thp_losses.append(thp_loss.item())
                val_thp_nll_losses.append(thp_nll_loss.item())
                val_thp_type_losses.append(thp_type_loss.item())
                val_thp_time_losses.append(thp_time_loss.item())
            else:
                loss = gen_loss + disc_loss
                val_thp_losses.append(0.0)
                val_thp_nll_losses.append(0.0)
                val_thp_type_losses.append(0.0)
                val_thp_time_losses.append(0.0)
            
            val_losses.append(loss.item())
            val_gen_losses.append(gen_loss.item())
            val_disc_losses.append(disc_loss.item())

    # Gather and average losses across all ranks
    val_losses = accelerator.gather(torch.tensor(val_losses, device=accelerator.device))
    val_gen_losses = accelerator.gather(torch.tensor(val_gen_losses, device=accelerator.device))
    val_disc_losses = accelerator.gather(torch.tensor(val_disc_losses, device=accelerator.device))
    val_thp_losses = accelerator.gather(torch.tensor(val_thp_losses, device=accelerator.device))
    val_thp_nll_losses = accelerator.gather(torch.tensor(val_thp_nll_losses, device=accelerator.device))
    val_thp_type_losses = accelerator.gather(torch.tensor(val_thp_type_losses, device=accelerator.device))
    val_thp_time_losses = accelerator.gather(torch.tensor(val_thp_time_losses, device=accelerator.device))
    
    # Compute mean on main process, keep as tensor for broadcast
    if accelerator.is_main_process:
        curr_val_loss = val_losses.mean()
        curr_val_gen_loss = val_gen_losses.mean()
        curr_val_disc_loss = val_disc_losses.mean()
        curr_val_thp_loss = val_thp_losses.mean()
        curr_val_thp_nll_loss = val_thp_nll_losses.mean()
        curr_val_thp_type_loss = val_thp_type_losses.mean()
        curr_val_thp_time_loss = val_thp_time_losses.mean()
    else:
        # Create dummy tensors on other ranks
        curr_val_loss = torch.tensor(0.0, device=accelerator.device)
        curr_val_gen_loss = torch.tensor(0.0, device=accelerator.device)
        curr_val_disc_loss = torch.tensor(0.0, device=accelerator.device)
        curr_val_thp_loss = torch.tensor(0.0, device=accelerator.device)
        curr_val_thp_nll_loss = torch.tensor(0.0, device=accelerator.device)
        curr_val_thp_type_loss = torch.tensor(0.0, device=accelerator.device)
        curr_val_thp_time_loss = torch.tensor(0.0, device=accelerator.device)
    
    # Broadcast tensors from main process to all ranks, then convert to float
    curr_val_loss = broadcast(curr_val_loss, from_process=0).item()
    curr_val_gen_loss = broadcast(curr_val_gen_loss, from_process=0).item()
    curr_val_disc_loss = broadcast(curr_val_disc_loss, from_process=0).item()
    curr_val_thp_loss = broadcast(curr_val_thp_loss, from_process=0).item()
    curr_val_thp_nll_loss = broadcast(curr_val_thp_nll_loss, from_process=0).item()
    curr_val_thp_type_loss = broadcast(curr_val_thp_type_loss, from_process=0).item()
    curr_val_thp_time_loss = broadcast(curr_val_thp_time_loss, from_process=0).item()
    
    return {
        'Optimization_Loss': curr_val_loss,
        'Generator_Loss': curr_val_gen_loss,
        'Discriminator_Loss': curr_val_disc_loss,
        'THP_Loss': curr_val_thp_loss,
        'THP_NLL_Loss': curr_val_thp_nll_loss,
        'THP_Type_Loss': curr_val_thp_type_loss,
        'THP_Time_Loss': curr_val_thp_time_loss
    }


def calculate_finetuning_eval_metrics(
    accelerator: Accelerator,
    task: str,
    predictions: np.ndarray, 
    targets: np.ndarray, 
    losses: np.ndarray,
    prefix: str = '',
    global_step: int = 0,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
) -> Tuple:
    """Calculate evaluation metrics for different prediction tasks."""

    if task == 'mortality':
        if np.sum(np.isnan(predictions)) > 0:
            print('Predictions contain NaN. Skipped validation and logging.')
            eval_output = {
                'Loss_Cross_Entropy': np.inf,
                'Accuracy': np.nan,
                'AUROC': np.nan,
                'AUPRC': np.nan,
                'F1_Score': np.nan
            }
        else:
            predicted_classes = np.array(predictions > 0.5, dtype=float)
            acc = accuracy_score(targets, predicted_classes)
            precision, recall, _ = precision_recall_curve(targets, predictions)
            auroc = roc_auc_score(targets, predictions)
            auprc = auc(recall, precision)
            f1 = f1_score(targets, predicted_classes)
            eval_output = {
                'Loss_Cross_Entropy': np.mean(losses),
                'Accuracy': acc,
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1_Score': f1
            }
            if writer is not None and accelerator.is_main_process:
                writer.add_scalars('Loss_Cross_Entropy', {prefix: np.mean(losses)}, global_step=global_step)
                writer.add_scalars('Accuracy', {prefix: acc}, global_step=global_step)
                writer.add_scalars('AUROC', {prefix: auroc}, global_step=global_step)
                writer.add_scalars('AUPRC', {prefix: auprc}, global_step=global_step)
                writer.add_scalars('F1_Score', {prefix: f1}, global_step=global_step)

    elif task == 'length_of_stay':
        mad = mean_absolute_error(targets, predictions)
        eval_output = {
            'Loss_Mean_Squared_Error': np.mean(losses),
            'Mean_Absolute_Error': mad
        }
        if writer is not None and accelerator.is_main_process:
            writer.add_scalars('Loss_Mean_Squared_Error', {prefix: np.mean(losses)}, global_step=global_step)
            writer.add_scalars('Mean_Absolute_Error', {prefix: mad}, global_step=global_step)

    elif task == 'phenotype':
        if np.sum(np.isnan(predictions)) > 0:
            print('Predictions contain NaN. Skipped validation and logging.')
            eval_output = {
                'Loss_Cross_Entropy': np.inf,
                'Micro_averaged_AUROC': np.nan,
                'Macro_averaged_AUROC': np.nan
            }
        else:
            micro_auroc = roc_auc_score(targets, predictions, average='micro')
            macro_auroc = roc_auc_score(targets, predictions, average='macro')
            eval_output = {
                'Loss_Cross_Entropy': np.mean(losses),
                'Micro_averaged_AUROC': micro_auroc,
                'Macro_averaged_AUROC': macro_auroc
            }
            if writer is not None and accelerator.is_main_process:
                writer.add_scalars('Loss_Cross_Entropy', {prefix: np.mean(losses)}, global_step=global_step)
                writer.add_scalars(f'Micro_averaged_AUROC', {prefix: micro_auroc}, global_step=global_step)
                writer.add_scalars(f'Macro_averaged_AUROC', {prefix: macro_auroc}, global_step=global_step)
    
    else:
        raise ValueError(f'task: Expected one of "mortality", "length_of_stay", or "phenotype", got {task}')

    return eval_output


def evaluate_finetuned_model(
    model: MixedClassifier,
    loader: torch.utils.data.DataLoader,
    task: str,
    accelerator: Accelerator,
    prefix: str = '',
    global_step: int = 0,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
) -> Dict[str, Any]:
    """Evaluates model performance for different tasks with Accelerate support.

    Args:
        model: The model to evaluate
        loader: DataLoader containing evaluation data
        task: One of 'mortality', 'length_of_stay', or 'phenotype'
        accelerator: Accelerator instance for distributed evaluation
        prefix: Prefix for TensorBoard logging
        global_step: Current training step
        writer: TensorBoard SummaryWriter object for logging

    Returns:
        Task-specific losses and performance metrics
    """
    if task == 'los':
        task = 'length_of_stay'

    if task not in ['mortality', 'length_of_stay', 'phenotype']:
        raise ValueError(f'task: Expected one of "mortality", "length_of_stay", or "phenotype", got {task}')
    
    model.eval()

    # Accumulate predictions, targets, and losses locally on each rank
    val_preds = []
    val_targs = []
    val_losses = []
    
    
    with torch.no_grad():
        desc = f'Evaluating {task} model performance'
        disable = not accelerator.is_local_main_process

        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False, disable=disable)):
            
            # Prepare input tensors
            # batch = prepare_input_tensors(batch, device=accelerator.device)
            logits = model(batch)

            # Calculate and store loss and predictions
            targets = batch['targets'][task]
            if task == 'mortality' or task == 'phenotype':
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                preds = torch.sigmoid(logits).detach()
            else:
                loss = torch.nn.functional.mse_loss(logits, targets)
                preds = logits.detach()

            # Store locally without gathering (avoids per-batch collective operations)
            val_losses.append(loss.detach())
            val_preds.append(preds)
            val_targs.append(targets.detach())

            # Explicitly delete large tensors and free cache
            del logits, batch
            torch.cuda.empty_cache()

        # Concatenate all local results
        if len(val_preds) > 0:
            val_losses = torch.stack(val_losses, dim=0)
            val_preds = torch.cat(val_preds, dim=0)
            val_targs = torch.cat(val_targs, dim=0)
        else:
            # Handle edge case where a rank has no data
            if task == 'mortality':
                output_dim = 1
            elif task == 'length_of_stay':
                output_dim = 1
            else:  # phenotype
                output_dim = targets.shape[-1] if 'targets' in locals() else 1
            val_losses = torch.empty(0)
            val_preds = torch.empty(0, output_dim)
            val_targs = torch.empty(0, output_dim)
            
        
        # Single gather operation after loop completes (synchronizes all ranks)
        val_losses = accelerator.gather(val_losses)
        val_preds = accelerator.gather(val_preds)
        val_targs = accelerator.gather(val_targs)
        
        # Only main process computes metrics
        if accelerator.is_main_process:            
            # If task == 'mortality', eval_ouput is (mean CE loss, accuracy, AUROC, AUPRC, F1).
            # If task == 'length_of_stay', eval_output is (MSE loss, mean absolute difference)
            # If task == 'phenotyping', eval_output is (mean CE loss, micro AUROC, macro AUROC)
            # calculate_eval_metrics also updates the TensorBoard SummaryWriter
            eval_output = calculate_finetuning_eval_metrics(
                accelerator=accelerator,
                task=task,
                predictions=val_preds.cpu().numpy(),
                targets=val_targs.cpu().numpy(),
                losses=val_losses.cpu().numpy(),
                prefix=prefix,
                global_step=global_step,
                writer=writer
            )
        else:
            eval_output = None
        eval_output = broadcast_object_list([eval_output], from_process=0)[0]
    
    # Wait for all processes to complete evaluation
    accelerator.wait_for_everyone()

    model.train()

    return eval_output


def pretrain_model(
    model: torch.nn.Module,
    save_path: str,
    loaders: Tuple[torch.utils.data.DataLoader],
    writer: torch.utils.tensorboard.SummaryWriter,
    learning_rate: float,
    learning_rate_decay: float = 0.5,
    total_epoch: int = 100,
    disc_loss_weight: float = 0.5,
    thp_loss_nll_weight: float = 1e-3,
    thp_loss_mc_samples: int = 100,
    use_thp_pred_loss: bool = True,
    thp_pred_loss_type_wt: float = 1.0,
    thp_pred_loss_time_wt: float = 0.01,
    record_mask_ratio: float = 0.25,
    obs_unobs_sample_ratio: float = 4.0,
    cmpnt_mask_ratio: float = 0.5,
    checkpoint_dir: str = None,
    resume_from_checkpoint: bool = True,
    timer: Optional[DistributedTimer] = None,
    accelerator: Accelerator = None,
    mem_test_mode: bool = False
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pre-train an ELECTRA-style model with Accelerate support (FSDP or DDP).
    
    Args:
        model: The ELECTRA model to pre-train
        save_path: Path to save the best model state dict
        loaders: Tuple of (train, val) or (train, val, test) DataLoaders
        writer: TensorBoard writer for logging (only used on main process)
        learning_rate: Learning rate for optimizer
        learning_rate_decay: Exponential decay factor every 50 epochs
        total_epoch: Number of training epochs
        disc_loss_weight: Weight for discriminator loss
        thp_loss_nll_weight: Weight for THP NLL loss
        thp_loss_mc_samples: Number of Monte Carlo samples for THP loss
        use_thp_pred_loss: Whether to include THP prediction losses
        thp_pred_loss_type_wt: Weight for THP type prediction loss
        thp_pred_loss_time_wt: Weight for THP time prediction loss
        record_mask_ratio: Fraction of timesteps to mask
        obs_unobs_sample_ratio: Ratio of observed to unobserved records
        cmpnt_mask_ratio: Fraction of components to mask in vector features
        checkpoint_dir: Directory for saving checkpoints
        resume_from_checkpoint: Whether to resume from checkpoint
        timer: Timer for tracking training time
        accelerator: Accelerator instance for distributed training
        mem_test_mode: If True, runs the forward and backward passes on a single batch, reports memory usage, and 
            raises an Exception to terminate. Useful for figuring out batch size limits.
    
    Returns:
        Tuple of (best_train_losses, best_val_losses) dictionaries
    """

    report_freq = 5
    chkpt_freq = 5

    if accelerator is None:
        raise ValueError("accelerator must be provided for distributed training")

    # Log distributed mode
    if accelerator.is_main_process:
        if is_fsdp_enabled(accelerator):
            print("Using FSDP for distributed training")
        else:
            print("Using DDP for distributed training")

    if len(loaders) == 2:
        train_loader, val_loader = loaders
    elif len(loaders) == 3:
        train_loader, val_loader, _ = loaders
    else:
        raise ValueError(f'`loaders` must be a tuple with 2 (train, val) or 3 members (train, val, test)')


    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    # Initialize loss functions
    gen_loss_fn = MaskedGeneratorLoss()
    disc_loss_fn = MaskedDiscriminatorLoss(weight=disc_loss_weight)
    thp_loss_fn = TransformerHawkesLoss(
        add_prediction_loss=use_thp_pred_loss,
        nll_weight=thp_loss_nll_weight,
        type_weight=thp_pred_loss_type_wt,
        time_weight=thp_pred_loss_time_wt
    )

    if accelerator.is_main_process:
        all_param_shapes = get_param_shapes(model)
        # Filter out LLM parameters
        param_shapes = OrderedDict()
        for key, shape in all_param_shapes.items():
            if not key.startswith('llm_module.'):
                param_shapes[key] = shape
    else:
        param_shapes = None
    param_shapes = broadcast_object_list([param_shapes], from_process=0)[0]

    # Prepare model, optimizer, and scheduler with Accelerate (handles both FSDP and DDP)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.wait_for_everyone()  # Ensure model is fully prepared before dataloaders - reduces peak memory usage

    # Prepare parallel dataloaders with Accelerate
    train_loader = accelerator.prepare_data_loader(train_loader)
    val_loader = accelerator.prepare_data_loader(val_loader)
    
    # Set up checkpoint file paths
    if checkpoint_dir is not None:
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator.wait_for_everyone()  # Ensure directory is created before other ranks proceed
        checkpoint_path = os.path.join(checkpoint_dir, 'pretrain_checkpoint')
        metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
        best_state_path = os.path.join(checkpoint_dir, 'best_model_state.pt')
    else:
        checkpoint_path = None
        metadata_path = None
        best_state_path = None

    # Initialize training metadata
    training_metadata = {}
    # Initialize best state dict
    best_state_dict = None


    # Load checkpoint if resuming
    if resume_from_checkpoint:
        training_metadata, best_state_dict = load_model_checkpoint(
            accelerator, checkpoint_path, metadata_path, best_state_path, timer
        )

    # If one of checkpointed sharded state, unsharded best state, or metadata was missing, start fresh
    start_epoch = training_metadata.get('start_epoch', 0)
    best_epoch = training_metadata.get('best_epoch', -1)
    best_epoch_train_losses = training_metadata.get('best_epoch_train_losses', {
        'Optimization_Loss': np.inf,
        'Generator_Loss': np.inf,
        'Discriminator_Loss': np.inf,
        'THP_Loss': np.inf,
        'THP_NLL_Loss': np.inf,
        'THP_Type_Loss': np.inf,
        'THP_Time_Loss': np.inf
    })
    best_epoch_val_losses = training_metadata.get('best_epoch_val_losses', {
        'Optimization_Loss': np.inf,
        'Generator_Loss': np.inf,
        'Discriminator_Loss': np.inf,
        'THP_Loss': np.inf,
        'THP_NLL_Loss': np.inf,
        'THP_Type_Loss': np.inf,
        'THP_Time_Loss': np.inf
    })
    best_epoch_val_loss = best_epoch_val_losses['Optimization_Loss']
    early_stopping_counter = training_metadata.get('early_stopping_counter', 0) 

    for epoch in tqdm(range(start_epoch, total_epoch), disable=not accelerator.is_local_main_process):

        # Training phase
        curr_epoch_train_losses = pretrain_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            gen_loss_fn=gen_loss_fn,
            disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio,
            accelerator=accelerator,
            desc=f"Epoch {epoch + 1} Training",
            mem_test_mode=mem_test_mode
        )

        # Validation phase
        curr_epoch_val_losses = pretrain_validate(
            model=model,
            loader=val_loader,
            gen_loss_fn=gen_loss_fn,
            disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio,
            accelerator=accelerator,
            desc=f"Epoch {epoch + 1} Validation"
        )

        # Logging
        if accelerator.is_main_process and writer is not None:
            for loss_name, loss_value in curr_epoch_train_losses.items():
                writer.add_scalar(f'{loss_name}/train', loss_value, epoch)
            for loss_name, loss_value in curr_epoch_val_losses.items():
                writer.add_scalar(f'{loss_name}/val', loss_value, epoch)


        # Check for improvement
        curr_val_loss = curr_epoch_val_losses['Optimization_Loss']
        improvement_made = curr_val_loss < best_epoch_val_loss


        # Update best losses and state dict if improvement made
        if improvement_made:
            best_epoch = epoch
            best_epoch_val_loss = curr_val_loss
            best_epoch_train_losses = curr_epoch_train_losses
            best_epoch_val_losses = curr_epoch_val_losses

            # Extract state dict (handles both FSDP and DDP)
            best_state_dict = extract_state_dict(accelerator, model, param_shapes)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        accelerator.wait_for_everyone()  # Ensure best_state_dict is ready on main process


        # Save current state (sharded), best state (unsharded), and training metadata every chkpt_freq epochs
        if checkpoint_dir is not None and (epoch + 1) % chkpt_freq == 0:

            save_model_checkpoint(
                accelerator=accelerator,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                epoch=epoch,
                best_epoch=best_epoch,
                best_epoch_train_losses=best_epoch_train_losses,
                best_epoch_val_losses=best_epoch_val_losses,
                early_stopping_counter=0,
                best_state_dict=best_state_dict,
                best_state_path=best_state_path,
                timer=timer
            )


        # Print progress
        if accelerator.is_main_process:
            if (epoch == 0) or ((epoch + 1) % report_freq == 0):
                performance_table = format_pretraining_performance_table(
                    epoch=epoch + 1,
                    current_train_losses=curr_epoch_train_losses,
                    current_val_losses=curr_epoch_val_losses,
                    best_train_losses=best_epoch_train_losses,
                    best_val_losses=best_epoch_val_losses,
                    use_thp_pred_loss=use_thp_pred_loss
                )
                print("\n" + performance_table + "\n")
        

        # Early stopping check
        if early_stopping_counter == 30:
            if accelerator.is_main_process:
                print(f"\nNo improvement observed within {early_stopping_counter} epochs. Stopping early.\n")
            break


        if (epoch + 1) % 50 == 0:
            scheduler.step()
    

    # Save final best state and encoder weights to model directory
    if accelerator.is_main_process:
        # Save full model params (unsharded)
        model_dir = os.path.dirname(save_path)
        os.makedirs(model_dir, exist_ok=True)
        # Save CPU-offloaded best state dict
        torch.save(best_state_dict, save_path)
        print(f"Saved pretrained model weights to {save_path}")

    # Save encoder weights separately for loading during finetuning
    save_encoder_weights(
        accelerator=accelerator,
        best_state_dict=best_state_dict,
        save_dir=os.path.dirname(save_path)
    )

    # Clean up checkpoint directory
    if accelerator.is_main_process:
        # Remove checkpoints
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        if metadata_path is not None and os.path.exists(metadata_path):
            os.remove(metadata_path)
        if best_state_path is not None and os.path.exists(best_state_path):
            os.remove(best_state_path)
    accelerator.wait_for_everyone()

    # Report peak GPU VRAM usage
    print_peak_memory(accelerator)

    return best_epoch_train_losses, best_epoch_val_losses


def finetune_model(
    model: MixedClassifier,
    save_path: str,
    loaders: Tuple[torch.utils.data.DataLoader],
    task: str,
    writer: torch.utils.tensorboard.SummaryWriter,
    learning_rate: float,
    learning_rate_decay: float = 0.8,
    total_epoch: int = 100,
    checkpoint_dir: str = None,
    resume_from_checkpoint: bool = True,
    timer: Optional[DistributedTimer] = None,
    accelerator: Accelerator = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fine-tune a pretrained model with Accelerate support (FSDP or DDP).
    
    Args:
        model: The pretrained model to fine-tune
        save_path: Path to save the best model
        loaders: Tuple of (train, val) or (train, val, test) DataLoaders
        task: One of 'mortality', 'length_of_stay', or 'phenotype'
        writer: TensorBoard writer (only used on main process)
        learning_rate: Learning rate for optimizer
        learning_rate_decay: Learning rate decay factor every 20 epochs
        total_epoch: Number of training epochs
        checkpoint_dir: Directory for saving checkpoints
        resume_from_checkpoint: Whether to resume from checkpoint
        timer: Timer for tracking training time
        accelerator: Accelerator instance for distributed training
    
    Returns:
        Tuple of (best_train_scores, best_val_scores) dictionaries
    """

    chkpt_freq = 5

    if accelerator is None:
        raise ValueError("accelerator must be provided for distributed training")

    # Log distributed mode
    if accelerator.is_main_process:
        if is_fsdp_enabled(accelerator):
            print("Using FSDP for distributed training")
        else:
            print("Using DDP for distributed training")

    if task == 'los':
        task = 'length_of_stay'
    if task not in ['mortality', 'length_of_stay', 'phenotype']:
        raise ValueError(f'task: Expected one of "mortality", "length_of_stay", or "phenotype", got {task}')

    if len(loaders) == 2:
        train_loader, val_loader = loaders
    elif len(loaders) == 3:
        train_loader, val_loader, _ = loaders
    else:
        raise ValueError(f'`loaders` must be a tuple with 2 (train, val) or 3 members (train, val, test)')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    if accelerator.is_main_process:
        all_param_shapes = get_param_shapes(model)
        # Filter out LLM parameters
        param_shapes = OrderedDict()
        for key, shape in all_param_shapes.items():
            if not key.startswith('llm_module.'):
                param_shapes[key] = shape
    else:
        param_shapes = None
    param_shapes = broadcast_object_list([param_shapes], from_process=0)[0]

    # Prepare model, optimizer, and dataloaders with Accelerate (handles both FSDP and DDP)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.wait_for_everyone()

    # Ensure model is fully prepared before dataloaders - reduces peak memory usage
    train_loader = accelerator.prepare_data_loader(train_loader)
    val_loader = accelerator.prepare_data_loader(val_loader)

    # Set up checkpoint file paths
    if checkpoint_dir is not None:
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator.wait_for_everyone()  # Ensure directory is created before other ranks proceed
        checkpoint_path = os.path.join(checkpoint_dir, f'{task}_checkpoint')
        metadata_path = os.path.join(checkpoint_dir, f'{task}_training_metadata.json')
        best_state_path = os.path.join(checkpoint_dir, f'{task}_best_model_state.pt')
    else:
        checkpoint_path = None
        metadata_path = None
        best_state_path = None
    
    # Initialize training metadata
    training_metadata = {}

    # Initialize best state dict
    best_state_dict = None    

    # Load checkpoint if resuming
    if resume_from_checkpoint:
        training_metadata, best_state_dict = load_model_checkpoint(
            accelerator, checkpoint_path, metadata_path, best_state_path, timer
        )
    # If one of checkpointed sharded state, unsharded best state, or metadata was missing, start fresh
    start_epoch = training_metadata.get('start_epoch', 0)
    best_epoch = training_metadata.get('best_epoch', -1)
    best_epoch_val_metric = training_metadata.get('best_epoch_val_metric', np.inf)
    best_epoch_train_scores = training_metadata.get('best_epoch_train_scores', None)
    best_epoch_val_scores = training_metadata.get('best_epoch_val_scores', None)
    early_stopping_counter = training_metadata.get('early_stopping_counter', 0)

    for epoch in tqdm(range(start_epoch, total_epoch), disable=not accelerator.is_local_main_process):

        # Training phase
        model.train()

        train_preds = []
        train_targs = []
        train_losses = []

        desc = f"Epoch {epoch + 1}, Training"
        disable = not accelerator.is_local_main_process
        for batch in tqdm(train_loader, desc=desc, leave=False, disable=disable):

            # batch = prepare_input_tensors(batch, device=accelerator.device)
            logits = model(batch)
        
            targets = batch['targets'][task]
            if task == 'mortality' or task == 'phenotype':
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                preds = torch.sigmoid(logits).detach()
            else:
                loss = torch.nn.functional.mse_loss(logits, targets)
                preds = logits.detach()

            # Store locally without gathering (avoids per-batch collective operations)
            train_losses.append(loss.detach())
            train_preds.append(preds)
            train_targs.append(targets)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        # Concatenate all local results
        if len(train_preds) > 0:
            train_losses = torch.stack(train_losses, dim=0)
            train_preds = torch.cat(train_preds, dim=0)
            train_targs = torch.cat(train_targs, dim=0)
        else:
            # Handle edge case where a rank has no data
            if task == 'mortality':
                output_dim = 1
            elif task == 'length_of_stay':
                output_dim = 1
            else:  # phenotype
                output_dim = train_targs.shape[-1] if len(train_targs) > 0 else 1
            train_losses = torch.empty(0)
            train_preds = torch.empty(0, output_dim)
            train_targs = torch.empty(0, output_dim)
        
        # Gather after loop
        train_losses = accelerator.gather(train_losses)
        train_preds = accelerator.gather(train_preds)
        train_targs = accelerator.gather(train_targs)

        if accelerator.is_main_process:
            current_epoch_train_scores = calculate_finetuning_eval_metrics(
                accelerator=accelerator,
                task=task,
                predictions=train_preds.cpu().numpy(),
                targets=train_targs.cpu().numpy(),
                losses=train_losses.cpu().numpy(),
                prefix='train',
                global_step=(epoch + 1),
                writer=writer
            )
        else:
            current_epoch_train_scores = None
        # Broadcast scores to all ranks from rank 0
        current_epoch_train_scores = broadcast_object_list([current_epoch_train_scores], from_process=0)[0]

        # Delete training losses, predictions, and targets to free memory
        del train_losses, train_preds, train_targs
        accelerator.free_memory()

        # Evaluate on validation set
        current_epoch_val_scores = evaluate_finetuned_model(
            model=model,
            loader=val_loader,
            task=task,
            accelerator=accelerator,
            global_step=(epoch + 1),
            prefix='val',
            writer=writer
        )

        # Check for improvement
        if task == 'mortality' or task == 'phenotype':
            current_epoch_val_metric = current_epoch_val_scores['Loss_Cross_Entropy']
        else:
            current_epoch_val_metric = current_epoch_val_scores['Loss_Mean_Squared_Error']
        improvement_made = current_epoch_val_metric < best_epoch_val_metric

        if improvement_made:
            best_epoch = epoch
            best_epoch_train_scores = current_epoch_train_scores
            best_epoch_val_metric = current_epoch_val_metric
            best_epoch_val_scores = current_epoch_val_scores
            early_stopping_counter = 0

            # Extract state dict (handles both FSDP and DDP)
            best_state_dict = extract_state_dict(accelerator, model, param_shapes)
            accelerator.wait_for_everyone()  # Ensure best_state_dict is ready on main process
        else:
            early_stopping_counter += 1

        # Save current state (sharded), best state (unsharded), and training metadata every chkpt_freq epochs
        if checkpoint_dir is not None and (epoch + 1) % chkpt_freq == 0:

            save_model_checkpoint(
                accelerator=accelerator,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                epoch=epoch,
                best_epoch=best_epoch,
                best_epoch_train_scores=best_epoch_train_scores,
                best_epoch_val_scores=best_epoch_val_scores,
                best_epoch_val_metric=best_epoch_val_metric,
                early_stopping_counter=early_stopping_counter,
                best_state_dict=best_state_dict,
                best_state_path=best_state_path,
                timer=timer,
                task=task
            )

        # Early stopping check
        if early_stopping_counter == 30:
            if accelerator.is_main_process:
                print(f"\nNo improvement observed within {early_stopping_counter} epochs. Stopping early.\n")
            break


        if (epoch + 1) % 20 == 0:
            scheduler.step()

    # Save best model
    if accelerator.is_main_process:
        # Save full model params (unsharded)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state_dict, save_path)
        print(f"Saved finetuned {task} model weights to {save_path}\n")

        # Remove checkpoints
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        if metadata_path is not None and os.path.exists(metadata_path):
            os.remove(metadata_path)
        if best_state_path is not None and os.path.exists(best_state_path):
            os.remove(best_state_path)
    
    accelerator.wait_for_everyone()

    # Report peak GPU VRAM usage
    print_peak_memory(accelerator)
    
    return best_epoch_train_scores, best_epoch_val_scores


def pretrain_with_hyperparameter(
    hp_name: str,
    hp_value: Any,
    model: torch.nn.Module,
    loaders: Tuple[torch.utils.data.DataLoader],
    writer: torch.utils.tensorboard.SummaryWriter,
    learning_rate: float,
    learning_rate_decay: float = 0.5,
    total_epoch: int = 100,
    disc_loss_weight: float = 0.5,
    thp_loss_nll_weight: float = 1e-3,
    thp_loss_mc_samples: int = 100,
    use_thp_pred_loss: bool = True,
    thp_pred_loss_type_wt: float = 1.0,
    thp_pred_loss_time_wt: float = 0.01,
    record_mask_ratio: float = 0.25,
    obs_unobs_sample_ratio: float = 4.0,
    cmpnt_mask_ratio: float = 0.5,
    checkpoint_dir: str = None,
    resume_from_checkpoint: bool = True,
    timer: Optional[DistributedTimer] = None,
    accelerator: Accelerator = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pretrain a model using a specific hyperparameter setting with Accelerate.
    
    This is the hyperparameter tuning version of pretrain_model.
    
    Args:
        hp_name: Name of the hyperparameter being tuned
        hp_value: Value of the hyperparameter being tested
        model: The ELECTRA model to pre-train
        loaders: Tuple of DataLoaders (train, test) or (train, val, test)
        writer: TensorBoard writer (only used on main process)
        learning_rate: Learning rate for optimizer
        learning_rate_decay: Exponential decay factor every 50 epochs
        total_epoch: Number of training epochs
        disc_loss_weight: Weight for discriminator loss
        thp_loss_nll_weight: Weight for THP NLL loss
        thp_loss_mc_samples: Number of Monte Carlo samples for THP loss
        use_thp_pred_loss: Whether to include THP prediction losses
        thp_pred_loss_type_wt: Weight for THP type prediction loss
        thp_pred_loss_time_wt: Weight for THP time prediction loss
        record_mask_ratio: Fraction of timesteps to mask
        obs_unobs_sample_ratio: Ratio of observed to unobserved records
        cmpnt_mask_ratio: Fraction of components to mask
        checkpoint_dir: Directory for saving checkpoints
        resume_from_checkpoint: Whether to resume from checkpoint
        timer: Timer for tracking training time
        accelerator: Accelerator instance for distributed training
    
    Returns:
        Tuple of (best_train_losses, best_val_losses) dictionaries
    """

    report_freq = 5
    chkpt_freq = 5


    if accelerator is None:
        raise ValueError("accelerator must be provided for distributed training")

    # Log distributed mode
    if accelerator.is_main_process:
        if is_fsdp_enabled(accelerator):
            print("Using FSDP for distributed training")
        else:
            print("Using DDP for distributed training")


    # For hyperparameter tuning, test set is used for model selection
    if len(loaders) == 2:
        train_loader, test_loader = loaders
    elif len(loaders) == 3:
        train_loader, _, test_loader = loaders
    else:
        raise ValueError(f'`loaders` must be a tuple with 2 (train, test) or 3 members (train, val, test)')
    

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)


    # Initialize loss functions
    gen_loss_fn = MaskedGeneratorLoss()
    disc_loss_fn = MaskedDiscriminatorLoss(weight=disc_loss_weight)
    thp_loss_fn = TransformerHawkesLoss(
        add_prediction_loss=use_thp_pred_loss,
        nll_weight=thp_loss_nll_weight,
        type_weight=thp_pred_loss_type_wt,
        time_weight=thp_pred_loss_time_wt
    )


    # Get the parameter shapes for unflattening later
    if accelerator.is_main_process:
        all_param_shapes = get_param_shapes(model)
        # Filter out LLM parameters
        param_shapes = OrderedDict()
        for key, shape in all_param_shapes.items():
            if not key.startswith('llm_module.'):
                param_shapes[key] = shape
    else:
        param_shapes = None
    param_shapes = broadcast_object_list([param_shapes], from_process=0)[0]    


    # Prepare model, optimizer, and scheduler with Accelerate (handles both FSDP and DDP)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.wait_for_everyone()  # Ensure model is fully prepared before dataloaders - reduces peak memory usage


    # Prepare parallel dataloaders with Accelerate
    train_loader = accelerator.prepare_data_loader(train_loader)
    test_loader = accelerator.prepare_data_loader(test_loader)


    # Set up checkpoint file paths
    if checkpoint_dir is not None:
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator.wait_for_everyone()  # Ensure directory is created before other ranks proceed
        checkpoint_path = os.path.join(checkpoint_dir, f'pretrain_checkpoint_{hp_name}_{hp_value}')
        metadata_path = os.path.join(checkpoint_dir, f'training_metadata_{hp_name}_{hp_value}.json')
        best_state_path = None  # Can set to a specific path if finetuning becomes part of the hypertuning workflow
    else:
        checkpoint_path = None
        metadata_path = None
        best_state_path = None


    # Initialize training metadata
    training_metadata = {}
    # Initialize best state dict
    best_state_dict = None


    # Load checkpoint if resuming
    if resume_from_checkpoint:
        training_metadata, _ = load_model_checkpoint(
            accelerator, checkpoint_path, metadata_path, best_state_path, timer
        )

    
    # If checkpointed sharded state or metadata was missing, start fresh
    start_epoch = training_metadata.get('start_epoch', 0)
    best_epoch = training_metadata.get('best_epoch', -1)
    best_epoch_train_losses = training_metadata.get('best_epoch_train_losses', {
        'Optimization_Loss': np.inf,
        'Generator_Loss': np.inf,
        'Discriminator_Loss': np.inf,
        'THP_Loss': np.inf,
        'THP_NLL_Loss': np.inf,
        'THP_Type_Loss': np.inf,
        'THP_Time_Loss': np.inf
    })
    best_epoch_val_losses = training_metadata.get('best_epoch_val_losses', {
        'Optimization_Loss': np.inf,
        'Generator_Loss': np.inf,
        'Discriminator_Loss': np.inf,
        'THP_Loss': np.inf,
        'THP_NLL_Loss': np.inf,
        'THP_Type_Loss': np.inf,
        'THP_Time_Loss': np.inf
    })
    best_epoch_val_loss = best_epoch_val_losses['Optimization_Loss']   
    early_stopping_counter = training_metadata.get('early_stopping_counter', 0) 


    for epoch in tqdm(range(start_epoch, total_epoch), disable=not accelerator.is_local_main_process):

        # Training phase
        curr_epoch_train_losses = pretrain_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            gen_loss_fn=gen_loss_fn,
            disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio,
            accelerator=accelerator,
            desc=f"Epoch {epoch + 1} Training"            
        )

        # Validation phase
        curr_epoch_val_losses = pretrain_validate(
            model=model,
            loader=test_loader,
            gen_loss_fn=gen_loss_fn,
            disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio,
            accelerator=accelerator,
            desc=f"Epoch {epoch + 1} Validation"
        )


        # Logging with TensorBoard
        if accelerator.is_main_process and writer is not None:
            for loss_name, loss_value in curr_epoch_train_losses.items():
                writer.add_scalars(f'{loss_name}/{hp_name}/train', {str(hp_value): loss_value}, epoch)
            for loss_name, loss_value in curr_epoch_val_losses.items():
                writer.add_scalars(f'{loss_name}/{hp_name}/val', {str(hp_value): loss_value}, epoch)
        

        # Check for improvement
        curr_epoch_val_loss = curr_epoch_val_losses['Optimization_Loss']
        improvement_made = curr_epoch_val_loss < best_epoch_val_loss


        # Update best losses if improvement made
        if improvement_made:
            best_epoch = epoch
            best_epoch_val_loss = curr_epoch_val_loss
            best_epoch_train_losses = curr_epoch_train_losses
            best_epoch_val_losses = curr_epoch_val_losses
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1


        # Get unsharded best state dict with immediate CPU offloading
        # NOTE This is only relevant if finetuning is part of hyperparameter tuning, and is commented out here.
        # best_state_dict = extract_state_dict(accelerator, model, param_shapes)
        # accelerator.wait_for_everyone()  # Ensure best_state_dict is ready on main process        


        # Save current state (sharded) and training metadata every chkpt_freq epochs
        if checkpoint_dir is not None and (epoch + 1) % chkpt_freq == 0:
            save_model_checkpoint(
                accelerator=accelerator,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                epoch=epoch,
                best_epoch_train_losses=best_epoch_train_losses,
                best_epoch_val_losses=best_epoch_val_losses,
                early_stopping_counter=early_stopping_counter,
                best_state_dict=best_state_dict,  # None if finetuning not part of HP tuning process
                best_state_path=best_state_path,  # None if finetuning not part of HP tuning process
                timer=timer
            )
        
        
        # Print progress
        if accelerator.is_main_process:
            if (epoch == 0) or ((epoch + 1) % report_freq == 0):
                performance_table = format_pretraining_performance_table(
                    epoch=epoch + 1,
                    current_train_losses=curr_epoch_train_losses,
                    current_val_losses=curr_epoch_val_losses,
                    best_train_losses=best_epoch_train_losses,
                    best_val_losses=best_epoch_val_losses,
                    use_thp_pred_loss=use_thp_pred_loss
                )
                print("\n" + performance_table + "\n")

        # Early stopping check
        if early_stopping_counter == 30:
            if accelerator.is_main_process:
                print(f"\nNo improvement observed within {early_stopping_counter} epochs. Stopping early.\n")
            break

        if (epoch + 1) % 50 == 0:
            scheduler.step()


    # Save final best state and encoder weights to model directory
    # NOTE This is only relevant if finetuning is part of the HP tuning process, and is commented out here
    # if accelerator.is_main_process:
    #     # Save full model params (unsharded)
    #     model_dir = os.path.dirname(save_path)
    #     os.makedirs(model_dir, exist_ok=True)
    #     # Save CPU-offloaded best state dict
    #     torch.save(best_state_dict, save_path)
    #     print(f"Saved pretrained model weights to {save_path}")


    # Save encoder weights separately for loading during finetuning
        # NOTE This is only relevant if finetuning is part of the HP tuning process, and is commented out here
    # save_encoder_weights(
    #     accelerator=accelerator,
    #     best_state_dict=best_state_dict,
    #     save_dir=os.path.dirname(save_path)
    # )


    # Clean up checkpoint directory
    if accelerator.is_main_process:
        # Remove checkpoints
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        if metadata_path is not None and os.path.exists(metadata_path):
            os.remove(metadata_path)
        if best_state_path is not None and os.path.exists(best_state_path):
            os.remove(best_state_path)
    accelerator.wait_for_everyone()

    # Report peak GPU VRAM usage
    print_peak_memory(accelerator)

    return best_epoch_train_losses, best_epoch_val_losses
