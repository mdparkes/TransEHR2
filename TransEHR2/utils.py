import numpy as np
import os
import time
import torch
import torch.distributed as dist
import yaml

from accelerate import Accelerator
from datetime import timedelta
from torch import Tensor
from typing import Any, Dict, List, OrderedDict, Tuple, Union

from TransEHR2.constants import TEXT_EMBED_DIM
from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.custom_types import MixedTensorDataset


class DistributedTimer:
    """Simplified timer for tracking pretraining times with checkpoint coordination."""
    
    def __init__(self, results_path: str = None):
        self.results_path = results_path
        self.world_size = 1
        self.rank = 0
        
        # Initialize distributed info if available
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_main_process = (self.rank == 0)
        
        self.times = {
            'total_start_time': None,
            'pretrain_total_time': 0.0,
            'finetune_total_time': 0.0,
            'most_recent_pretrain_time': 0.0,
            'most_recent_finetune_time': 0.0,
            'fold_times': {},
            'current_fold': None,
            'current_fold_start_time': None,
            'current_phase_start_time': None,
            'current_phase_elapsed': 0.0,
            'world_size': self.world_size
        }
    
    def start_total_timing(self):
        """Start timing the entire experiment."""
        if self.times['total_start_time'] is None:
            self.times['total_start_time'] = time.time()
    
    def start_fold(self, fold_name: str):
        """Start timing a specific fold."""
        self.times['current_fold'] = fold_name
        if fold_name not in self.times['fold_times']:
            self.times['fold_times'][fold_name] = {
                'pretrain_time': 0.0,
                'finetune_time': 0.0,
                'total_time': 0.0,
                'start_time': time.time()
            }
        self.times['current_fold_start_time'] = time.time()
    
    def start_phase(self, phase: str, is_main_process: bool):
        """Start timing a phase (pretrain/finetune)."""
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # If resuming from checkpoint, account for already elapsed time
        if self.times['current_phase_elapsed'] > 0:
            self.times['current_phase_start_time'] = time.time() - self.times['current_phase_elapsed']
            if is_main_process:
                print(f"Resuming {phase} phase with {self.times['current_phase_elapsed']:.1f}s already elapsed")
        else:
            self.times['current_phase_start_time'] = time.time()
    
    def end_phase(self, phase: str, is_main_process: bool):
        """End timing a phase and update totals."""
        if self.times['current_phase_start_time'] is None:
            return 0.0
        
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        elapsed = time.time() - self.times['current_phase_start_time']
        
        if phase == 'pretrain':
            self.times['pretrain_total_time'] += elapsed
            self.times['most_recent_pretrain_time'] = elapsed  # Overwrite with most recent
            if self.times['current_fold']:
                self.times['fold_times'][self.times['current_fold']]['pretrain_time'] += elapsed
        elif phase == 'finetune':
            self.times['finetune_total_time'] += elapsed
            self.times['most_recent_finetune_time'] = elapsed  # Overwrite with most recent
            if self.times['current_fold']:
                self.times['fold_times'][self.times['current_fold']]['finetune_time'] += elapsed
        
        # Reset phase tracking
        self.times['current_phase_start_time'] = None
        self.times['current_phase_elapsed'] = 0.0
        
        # Save results immediately
        if is_main_process:
            self.save_results(is_main_process)
        
        return elapsed
    
    def end_fold(self, is_main_process: bool):
        """End timing current fold."""
        if self.times['current_fold'] and self.times['current_fold_start_time']:
            fold_name = self.times['current_fold']
            total_fold_time = time.time() - self.times['current_fold_start_time']
            self.times['fold_times'][fold_name]['total_time'] = total_fold_time
            
            if is_main_process:
                print(f"\n{'='*60}")
                print(f"FOLD {fold_name.upper()} COMPLETED")
                print(f"{'='*60}")
                print(f"Most recent pretraining time: {self._format_time(self.times['most_recent_pretrain_time'])}")
                print(f"Most recent finetuning time: {self._format_time(self.times['most_recent_finetune_time'])}")
                print(f"Total fold time: {self._format_time(total_fold_time)}")
                if self.world_size > 1:
                    print(f"World size: {self.world_size} GPUs")
                print(f"{'='*60}\n")
        
        self.times['current_fold'] = None
        self.times['current_fold_start_time'] = None
        
        if is_main_process:
            self.save_results(is_main_process)
    
    def get_timer_state_for_checkpoint(self) -> dict:
        """Get timer state to include in model checkpoints."""
        if self.times['current_phase_start_time'] is not None:
            current_elapsed = time.time() - self.times['current_phase_start_time']
        else:
            current_elapsed = 0.0
            
        return {
            'timer_state': self.times.copy(),
            'current_phase_elapsed': current_elapsed
        }
    
    def restore_from_checkpoint(self, checkpoint_data: dict, is_main_process: bool):
        """Restore timer state from model checkpoint."""
        if 'timer_state' in checkpoint_data:
            timer_state = checkpoint_data['timer_state']
            current_elapsed = checkpoint_data.get('current_phase_elapsed', 0.0)
            
            self.times.update(timer_state)
            self.times['current_phase_elapsed'] = current_elapsed
            
            if is_main_process:
                print(f"Restored timer state from checkpoint")
                if current_elapsed > 0:
                    print(f"Will resume with {current_elapsed:.1f}s already elapsed")
    
    def get_total_time(self):
        """Get total experiment time."""
        if self.times['total_start_time']:
            return time.time() - self.times['total_start_time']
        return 0.0
    
    def print_final_summary(self, is_main_process: bool):
        """Print final timing summary."""
        if not is_main_process:
            return
        
        total_time = self.get_total_time()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT TIMING SUMMARY")
        if self.world_size > 1:
            print(f"Multi-GPU Training: {self.world_size} GPUs")
        print(f"{'='*80}")
        print(f"Total experiment time: {self._format_time(total_time)}")
        print(f"Total cumulative pretraining time: {self._format_time(self.times['pretrain_total_time'])}")
        print(f"Total cumulative finetuning time: {self._format_time(self.times['finetune_total_time'])}")
        print()
        print(f"Most recent model pretraining time: {self._format_time(self.times['most_recent_pretrain_time'])}")
        print(f"Most recent model finetuning time: {self._format_time(self.times['most_recent_finetune_time'])}")
        
        if self.world_size > 1:
            recent_pretrain_gpu_hours = self.times['most_recent_pretrain_time'] * self.world_size
            recent_finetune_gpu_hours = self.times['most_recent_finetune_time'] * self.world_size
            total_pretrain_gpu_hours = self.times['pretrain_total_time'] * self.world_size
            total_finetune_gpu_hours = self.times['finetune_total_time'] * self.world_size
            
            print(f"\nEffective compute time (GPU-hours):")
            print(f"  Most recent model pretraining: {self._format_time(recent_pretrain_gpu_hours)}")
            print(f"  Most recent model finetuning: {self._format_time(recent_finetune_gpu_hours)}")
            print(f"  Total cumulative pretraining: {self._format_time(total_pretrain_gpu_hours)}")
            print(f"  Total cumulative finetuning: {self._format_time(total_finetune_gpu_hours)}")
        
        if self.times['fold_times']:
            print(f"\nPer-fold breakdown:")
            print(f"{'-'*80}")
            for fold_name, fold_times in self.times['fold_times'].items():
                print(f"{fold_name:10} | "
                      f"Pretrain: {self._format_time(fold_times['pretrain_time']):>12} | "
                      f"Finetune: {self._format_time(fold_times['finetune_time']):>12} | "
                      f"Total: {self._format_time(fold_times['total_time']):>12}")
        
        print(f"{'='*80}\n")
        
        # Final save
        if is_main_process:
            self.save_results(is_main_process)
    
    def save_results(self, is_main_process: bool):
        """Save timing results to disk."""
        if not is_main_process or not self.results_path:
            return
            
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        results_data = self.times.copy()
        results_data['total_experiment_time'] = self.get_total_time()
        results_data['updated_at'] = time.time()
        results_data['updated_at_readable'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        with open(self.results_path, 'w') as f:
            yaml.dump(results_data, f, default_flow_style=False, indent=2)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        return str(timedelta(seconds=int(seconds)))

# Keep backward compatibility and add convenience function
Timer = DistributedTimer


def create_timer(results_dir: str = None, experiment_name: str = "experiment") -> DistributedTimer:
    """Create a timer with simplified settings."""
    
    results_path = None
    if results_dir:
        results_path = os.path.join(results_dir, f"{experiment_name}_timing_results.yaml")
    
    return DistributedTimer(results_path=results_path)


def ensure_float32(data: MixedDataset) -> MixedDataset:
    """Converts float64-valued tensors in `data` to float32.

    MPS doesn't support float64 tensors, so this function is used to ensure that all tensors in the dataset are float32.

    Args:
        data (MixedDataset): The dataset to be converted.
    
    Returns:
        MixedDataset: The converted dataset.
    """

    for i, tnsr in enumerate(data):
        if tnsr.dtype == torch.float64:
            data[i] = tnsr.float()
    return data


def get_non_pad_mask(seq: Tensor, padding_value: int = 0) -> Tensor:
    """Get the mask for non-padding items in the input sequence
    
    Given a predetermined maximum sequence length, this function returns a mask tensor that indicates which tokens
    in the sequence are *not* padding tokens.

    Args:
        seq (Tensor): A `Tensor` of shape [batch size, sequence length] containing the input sequence.
        padding_value (int, optional): The value of the padding tokens in `seq`. Defaults to 0.
    
    Returns:
        Tensor: A `Tensor` of shape [batch size, sequence length, 1] containing the mask for non-padding tokens.
    """

    assert seq.dim() == 2
    non_padding_mask = seq.ne(padding_value).type(torch.float32).unsqueeze(-1)  # ne: not equal 
    return non_padding_mask


def combine_value_and_text_data(
        value_assoc_indicators: Tensor,
        value_assoc_values: Tensor,
        text_assoc_indicators: Tensor,
        text_embeddings: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Concatenate the LLM-generated text embeddings with the value-associated data.
        
        This function combines all the value-associated indicators and values so that it can be passed as input to the 
        predictive module.

        Note:
            The `value_assoc_indicators` and `text_assoc_indicators` tensors must be aligned along the last dimension,
            meaning the feature order and batch/timestep alignment must be consistent for correct concatenation.

        Args:
            value_assoc_indicators: Tensor of shape (batch_size, max_timeseries_length, n_features) with indicators for 
                value-associated data.
            value_assoc_values: Tensor of shape (batch_size, max_timeseries_length, *) with value-associated 
                data, where * is the total number of numeric and categorical feature dimensions.
            text_assoc_indicators: Tensor of shape (batch_size, max_timeseries_length, n_text_features) for 
                text-associated indicators.
            text_embeddings: Tensor of shape (batch_size, max_timeseries_length, n_text_features, embedding_dim) 
                for text embeddings generated by the LLM.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Concatenated indicators of shape (batch_size, max_timeseries_length, total_num_features), where
                  `total_num_features` is the sum of the number of numeric, categorical, and text features.
                - Concatenated values of shape (batch_size, max_timeseries_length, total_feature_dim), where
                  `total_feature_dim` is the sum of numeric, categorical, and text feature dimensions.
        """
        
        # Concatenate value- and text-associated indicators
        indicators = torch.cat([value_assoc_indicators, text_assoc_indicators], dim=-1)
        # Concatenate the embeddings of the text features in the last dimension
        text_assoc_values = text_embeddings.flatten(start_dim=2)
        # Concatenate value- and text-associated vals
        values = torch.cat([value_assoc_values, text_assoc_values], dim=-1)

        return indicators, values


def generate_record_masks(
    data: MixedTensorDataset,
    feature_sample_rate: float = 0.15,
    obs_unobs_ratio: float = 4.0,
    subsample_rate: float = 0.5
) -> Tuple[Dict[str, Dict[str, Union[Tensor, List[Tensor]]]], Tensor]:
    """Generate masks for observed and unobserved records in the dataset.

    This function randomly selects records to mask in the dataset. It samples a fixed percentage of the observed records to mask and tries to sample a number of unobserved records such that the ratio of masked observed records to masked unobserved records satisfies a specified ratio (`obs_unobs_ratio`). The components of vector-valued features that were selected for masking are subsampled so that only a specified portion of the components are masked. The number of components to subsample is max(1, floor(subsample_rate * dim)), where `dim` is the number of components in the vector-valued feature. Masked components are represented as ones; unmasked components are represented as zeros.

    Args:
        data: Batched MixedTensorDataset from DataLoader containing value-associated and event-associated data.
        feature_sample_rate (float): The rate at which features are sampled for masking.
        obs_unobs_ratio (float): The ratio of observed to unobserved records that sampling will try to achieve.
        subsample_rate (float): The rate at which components of vector-valued features are subsampled for masking.

    Returns:
        Tuple[Dict[str, Dict[str, Union[Tensor, List[Tensor]]]], Tensor]: A tuple of masks for value-associated and event-associated data, respectively.
        Value-associated masks have the following structure:
        ```
        {
            'numeric': {
                'indicators': tensor(batch_size, max_ts_len, n_numeric_feats)  # Mask indicators for features
                'values': [  # List of features
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 1 components
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 2 components
                    ...  # More features
                ]
            },
            'categorical': {
                'indicators': tensor(batch_size, max_ts_len, n_cat_feats)  # Mask indicators for categorical features
                'values': [  # List of features
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 1 components
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 2 components
                    ...  # More features
                ]
            },
            'text': {
                'indicators': tensor(batch_size, max_ts_len, n_text_feats)  # Mask indicators for text features
                'embedded_values': [  # List of text features
                    tensor(batch_size, max_ts_len, TEXT_EMBED_DIM),  # Mask indicators for Feature 1 components
                    tensor(batch_size, max_ts_len, TEXT_EMBED_DIM),  # Mask indicators for Feature 2 components
                    ...  # More features
                ]
            } 
        ```

        Event-associated masks have the following structure:
        ```
        tensor(batch_size, max_ts_len, n_event_feats)  # Mask indicators for features
        ```
    """

    # Get batch dimensions from the collated data structure
    batch_size, max_ts_len = data['val_data']['times'].shape
    batch_device = data['val_data']['times'].device

    val_masks = {}
    event_masks = None

    # Initialize value-associated data masks
    for feature_type in ['numeric', 'categorical', 'text']:
        if feature_type in data['val_data']:
            feature_data = data['val_data'][feature_type]
            # Get number of features from first episode's first timestep
            n_features = feature_data['indicators'].shape[-1]  # features per timestep
            # Initialize indicator mask tensor
            indicator_mask = torch.zeros_like(feature_data['indicators'], device=batch_device)
            if feature_type == 'text':
                # Text features get embedded to TEXT_EMBED_DIM
                value_mask_shape = (batch_size, max_ts_len, TEXT_EMBED_DIM)
                val_masks[feature_type] = {
                    'indicators': indicator_mask,
                    'embedded_values': [torch.zeros(value_mask_shape, device=batch_device) for _ in range(n_features)]
                }
            else:
                # For numeric/categorical, we need to determine feature dimensions
                val_masks[feature_type] = {
                    'indicators': indicator_mask,
                    'values': [torch.zeros_like(tnsr, device=batch_device) for tnsr in feature_data['values']]
                }

    # Initialize event-associated masks
    if 'event_data' in data:
        event_masks = torch.zeros_like(data['event_data']['indicators'], device=batch_device)

    # Generate the value-associated masks
    for feature_type in ['numeric', 'categorical', 'text']:
        if feature_type in val_masks:
            _gen_val_assoc_feat_mask(
                data, feature_type, val_masks, feature_sample_rate, obs_unobs_ratio, subsample_rate
            )
    
    # Generate the event-associated data masks
    if event_masks is not None:
        _gen_event_assoc_feat_mask(
            data, event_masks, feature_sample_rate, obs_unobs_ratio
        )

    return val_masks, event_masks


def _gen_val_assoc_feat_mask(
    data: Dict, feature_type: str, val_masks: Dict, 
    feature_sample_rate: float, obs_unobs_ratio: float, subsample_rate: float
):
    """Generate value-associated feature masks for batch data.
    
    Feature masking is done over all batch instances combined.
    """

    indicators_data = data['val_data'][feature_type]['indicators']
    if feature_type != 'text':
        values_key = 'values'
        values_data = data['val_data'][feature_type][values_key]
    else:
        values_key = 'embedded_values'  # Key used in val_masks for text features
    padding_mask = data['val_data']['masks'].unsqueeze(-1)  # Shape: (batch_size, max_ts_len, 1)

    # Count the total number of observed and unobserved features in the whole batch
    obs_feats = (indicators_data == 1) & padding_mask.bool()
    unobs_feats = (indicators_data == 0) & padding_mask.bool()

    obs_positions = torch.nonzero(obs_feats, as_tuple=False)
    unobs_positions = torch.nonzero(unobs_feats, as_tuple=False)

    obs_count = obs_positions.size(0)
    unobs_count = unobs_positions.size(0)

    n_obs_masked = int(feature_sample_rate * obs_count)
    n_unobs_masked = min(unobs_count, max(1, int(n_obs_masked / obs_unobs_ratio))) if unobs_count > 0 else 0

    for positions, n_masked in [(obs_positions, n_obs_masked), (unobs_positions, n_unobs_masked)]:
        if n_masked > 0:
            perm = torch.randperm(positions.size(0))[:n_masked]
            selected = positions[perm]
            val_masks[feature_type]['indicators'][selected[:,0], selected[:,1], selected[:,2]] = 1.0
            # Iterate over selected positions to mask corresponding value components
            for idx in range(selected.size(0)):
                b, t, f = selected[idx]  # batch index, timestep, feature
                feat_dim = TEXT_EMBED_DIM if feature_type == 'text' else values_data[f].shape[-1]
                n_components_to_mask = max(1, int(subsample_rate * feat_dim))
                component_indices = torch.randperm(feat_dim)[:n_components_to_mask]
                val_masks[feature_type][values_key][f][b, t, component_indices] = 1.0

    return val_masks


def _gen_event_assoc_feat_mask(
    data: Dict, event_masks: Tensor, 
    feature_sample_rate: float, obs_unobs_ratio: float
):
    """Generate event-associated feature masks using efficient tensor operations."""

    indicators_data = data['event_data']['indicators']
    padding_mask = data['event_data']['masks'].unsqueeze(-1)

    # Create boolean masks for observed and unobserved features
    obs_feats = (indicators_data == 1) & padding_mask.bool()
    unobs_feats = (indicators_data == 0) & padding_mask.bool()

    # Get tensor positions (avoid .tolist() conversion)
    obs_positions = torch.nonzero(obs_feats, as_tuple=False)
    unobs_positions = torch.nonzero(unobs_feats, as_tuple=False)

    # Calculate the number of observed and unobserved features to mask
    n_obs_masked = int(feature_sample_rate * obs_positions.size(0))
    n_unobs_masked = min(
        unobs_positions.size(0), max(1, int(n_obs_masked / obs_unobs_ratio))
    ) if unobs_positions.size(0) > 0 else 0

    # Process both observed and unobserved positions efficiently
    for positions, n_masked in [(obs_positions, n_obs_masked), (unobs_positions, n_unobs_masked)]:
        if n_masked > 0:
            perm = torch.randperm(positions.size(0))[:n_masked]
            selected = positions[perm]
            # Vectorized assignment
            event_masks[selected[:,0], selected[:,1], selected[:,2]] = 1.0
    
    return event_masks


def format_pretraining_performance_table(
    epoch: int,
    current_train_losses: dict,
    current_val_losses: dict,
    best_train_losses: dict,
    best_val_losses: dict,
    use_thp_pred_loss: bool
) -> str:
    """Format an ASCII table for pretraining metrics."""
    
    # Fixed width calculation:
    # 37 (max description width) + 4 (spaces) + 17 (max value width) = 58
    # Total table width with borders: 58 + 2 = 60
    TABLE_WIDTH = 60
    CONTENT_WIDTH = 58
    DESC_WIDTH = 37  # Maximum description width
    VALUE_WIDTH = 17  # Maximum value width (12 digits + 1 decimal + 4 trailing)
    
    def format_value(value, is_string=False):
        """Format a value to occupy exactly 17 characters if ≤12 leading digits."""
        if is_string:
            return f"{value:>{VALUE_WIDTH}}"
        
        # Format the number with 4 decimal places
        formatted = f"{value:.4f}"
        
        # Find the decimal point position to count leading digits
        decimal_pos = formatted.find('.')
        leading_digits = decimal_pos
        
        # If 12 or fewer leading digits, pad to exactly 17 characters
        if leading_digits <= 12:
            padding_needed = max(0, 12 - leading_digits)
            padded_value = " " * padding_needed + formatted
            return f"{padded_value:>{VALUE_WIDTH}}"[:VALUE_WIDTH]
        else:
            # More than 12 leading digits - no padding, will misalign
            return formatted
    
    def format_loss_section(losses: dict, section_title: str) -> list:
        lines = []
        lines.append(f"│ {section_title:<{CONTENT_WIDTH}} │")
        lines.append("├" + "─" * TABLE_WIDTH + "┤")
        
        # Each line: description (37 chars) + 4 spaces + value (17 chars)
        lines.append(f"│ {'Mean total loss:':<{DESC_WIDTH}}    {format_value(losses['Optimization_Loss'])} │")
        lines.append(f"│ {'Generator:':<{DESC_WIDTH}}    {format_value(losses['Generator_Loss'])} │")
        lines.append(f"│ {'Discriminator:':<{DESC_WIDTH}}    {format_value(losses['Discriminator_Loss'])} │")
        lines.append(f"│ {'THP (overall, weighted):':<{DESC_WIDTH}}    {format_value(losses['THP_Loss'])} │")
        lines.append(f"│ {'THP neg. log likelihood (unweighted):':<{DESC_WIDTH}}    {format_value(losses['THP_NLL_Loss'])} │")
        
        # Handle THP prediction losses
        if use_thp_pred_loss:
            lines.append(f"│ {'THP event type loss (unweighted):':<{DESC_WIDTH}}    {format_value(losses['THP_Type_Loss'])} │")
            lines.append(f"│ {'THP time loss (unweighted):':<{DESC_WIDTH}}    {format_value(losses['THP_Time_Loss'])} │")
        else:
            lines.append(f"│ {'THP event type loss (unweighted):':<{DESC_WIDTH}}    {format_value('Not used', is_string=True)} │")
            lines.append(f"│ {'THP time loss (unweighted):':<{DESC_WIDTH}}    {format_value('Not used', is_string=True)} │")
        
        return lines
    
    # Build the complete table
    table_lines = []
    
    # Title
    title = f"Pretraining Performance, Epoch {epoch}"
    table_lines.append("┌" + "─" * TABLE_WIDTH + "┐")
    table_lines.append(f"│ {title:^{CONTENT_WIDTH}} │")
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Current epoch validation losses
    table_lines.extend(format_loss_section(current_val_losses, "Current Epoch Mean Losses (Validation)"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Current epoch training losses
    table_lines.extend(format_loss_section(current_train_losses, "Current Epoch Mean Losses (Training)"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Best epoch validation losses
    table_lines.extend(format_loss_section(best_val_losses, "Best Epoch Mean Losses (Validation)"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Best epoch training losses  
    table_lines.extend(format_loss_section(best_train_losses, "Best Epoch Mean Losses (Training)"))
    
    # Close table
    table_lines.append("└" + "─" * TABLE_WIDTH + "┘")
    
    return "\n".join(table_lines)


def format_finetuning_performance_table(
    task: str,
    train_scores: dict,
    val_scores: dict,
    test_scores: dict
) -> str:
    """Format an ASCII table for finetuning metrics."""
    
    # Fixed width calculation: same as pretraining table
    TABLE_WIDTH = 60
    CONTENT_WIDTH = 58
    DESC_WIDTH = 37  # Maximum description width
    VALUE_WIDTH = 17  # Maximum value width (12 digits + 1 decimal + 4 trailing)
    
    def format_value(value, is_string=False):
        """Format a value to occupy exactly 17 characters if ≤12 leading digits."""
        if is_string:
            return f"{value:>{VALUE_WIDTH}}"
        
        # Format the number with 4 decimal places
        formatted = f"{value:.4f}"
        
        # Find the decimal point position to count leading digits
        decimal_pos = formatted.find('.')
        leading_digits = decimal_pos
        
        # If 12 or fewer leading digits, pad to exactly 17 characters
        if leading_digits <= 12:
            padding_needed = max(0, 12 - leading_digits)
            padded_value = " " * padding_needed + formatted
            return f"{padded_value:>{VALUE_WIDTH}}"[:VALUE_WIDTH]
        else:
            # More than 12 leading digits - no padding, will misalign
            return formatted
    
    def format_score_section(scores: dict, section_title: str) -> list:
        lines = []
        lines.append(f"│ {section_title:<{CONTENT_WIDTH}} │")
        lines.append("├" + "─" * TABLE_WIDTH + "┤")
        
        if task == 'mortality':
            lines.append(f"│ {'Mean cross-entropy loss:':<{DESC_WIDTH}}    {format_value(scores['Loss_Cross_Entropy'])} │")
            lines.append(f"│ {'Accuracy:':<{DESC_WIDTH}}    {format_value(scores['Accuracy'])} │")
            lines.append(f"│ {'AUROC:':<{DESC_WIDTH}}    {format_value(scores['AUROC'])} │")
            lines.append(f"│ {'AUPRC:':<{DESC_WIDTH}}    {format_value(scores['AUPRC'])} │")
            lines.append(f"│ {'F1:':<{DESC_WIDTH}}    {format_value(scores['F1_Score'])} │")

        elif task == 'length_of_stay':
            # CORRECTED: Use the actual keys from calculate_finetuning_eval_metrics
            lines.append(f"│ {'Mean squared error loss:':<{DESC_WIDTH}}    {format_value(scores['Loss_Mean_Squared_Error'])} │")
            lines.append(f"│ {'Mean absolute difference:':<{DESC_WIDTH}}    {format_value(scores['Mean_Absolute_Error'])} │")

        else:  # phenotype
            lines.append(f"│ {'Mean cross-entropy loss:':<{DESC_WIDTH}}    {format_value(scores['Loss_Cross_Entropy'])} │")
            lines.append(f"│ {'Microaveraged AUROC:':<{DESC_WIDTH}}    {format_value(scores['Micro_averaged_AUROC'])} │")
            lines.append(f"│ {'Macroaveraged AUROC:':<{DESC_WIDTH}}    {format_value(scores['Macro_averaged_AUROC'])} │")
        
        return lines
    
    # Build the complete table
    table_lines = []
    
    # Title
    title = f"Finetuned {task} Model Performance"
    table_lines.append("┌" + "─" * TABLE_WIDTH + "┐")
    table_lines.append(f"│ {title:^{CONTENT_WIDTH}} │")
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Training set scores
    table_lines.extend(format_score_section(train_scores, "Training set"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Validation set scores
    table_lines.extend(format_score_section(val_scores, "Validation set"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Test set scores
    table_lines.extend(format_score_section(test_scores, "Test set"))
    
    # Close table
    table_lines.append("└" + "─" * TABLE_WIDTH + "┘")
    
    return "\n".join(table_lines)


def convert_to_python_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensor and numpy types in a dictionary to native Python types."""

    converted = {}
    
    loss_key_order = [
        'Optimization_Loss',
        'Generator_Loss', 
        'Discriminator_Loss',
        'THP_Loss',
        'THP_NLL_Loss',
        'THP_Type_Loss',
        'THP_Time_Loss'
    ]
    
    # Add keys in the specified order
    for key in loss_key_order:
        if key in data:
            val = data[key]
            if hasattr(val, 'item'):  # It's a tensor
                converted[key] = float(val.item())
            elif isinstance(val, (torch.Tensor, np.ndarray)):
                converted[key] = float(val)
            else:
                converted[key] = val
    
    # Add any remaining keys not in the specified order
    for key, val in data.items():
        if key not in converted:
            if hasattr(val, 'item'):  # It's a tensor
                converted[key] = float(val.item())
            elif isinstance(val, (torch.Tensor, np.ndarray)):
                converted[key] = float(val)
            else:
                converted[key] = val
                
    return converted


def calc_time_diff(event_times: Tensor, non_pad_mask: Tensor, device: str) -> Tensor:
        """Calculate time differences between consecutive events.
        
        Temporal differences are calculated between the current timestamp and the previous one.

        Args:
            event_times (Tensor): Event timestamps of shape (batch_size, max_ts_len)
            non_pad_mask (Tensor): Non-padding mask of shape (batch_size, max_ts_len)
        
        Returns:
            Tensor: Time differences of shape (batch_size, max_ts_len). The first timestep has a time difference of zero.
        
        """
        
        time_diff = torch.zeros_like(event_times, device=device)
        time_diff[:, 1:] = (event_times[:, 1:] - event_times[:, :-1]) * non_pad_mask[:, 1:]

        return time_diff


def sample_non_event_time_diff(time_diff_seq: Tensor, n: int, device: str) -> Tensor:
    """Sample non-event time differences using uniform distribution.

    Given a sequences of time differences between the time of the each observed event and the one preceding it, this function uniformly samples time differences in [0, time_diff). In other words, it uniformly samples timestamps between consecutive events and returns the difference between the sampled timestamp and the previous observed timestamp.

    Note that Zuo et al. used random sampling on the interval [0, time_diff), but this implementation uses evenly spaced sampling on the interval [0, time_diff], inclusive of the bounds. While intensities are technically undefined at the left side of the interval, it's okay to include the bounds of the interval because the sampled times are being used to estimate the integral of the intensity function over the interval, and including the bounds should improve the accuracy of the integral estimate without invalidating it.

    Args:
        n (int): Number of samples to draw
        time_diff_seq (Tensor): Time differences of shape (batch_size, max_ts_len)

    Returns:
        Tensor: Sampled time differences of shape (batch_size, max_ts_len, n_samples)
    """

    sampled_ratios = torch.linspace(0., 1., n, device=device)[None, None, :]  # (1, 1, n_samples)
    sampled_time_diffs = time_diff_seq[:, :, None] * sampled_ratios  # (batch_size, max_ts_len, n_samples)

    return sampled_time_diffs


def get_param_shapes(model: torch.nn.Module) -> OrderedDict[str, Tuple[int]]:
    """Get the expected shapes of all parameters and buffers in a model.
    
    This should be called BEFORE wrapping the model with FSDP, as FSDP will
    flatten parameters and make their shapes unreliable. The returned dictionary
    can then be used to reshape flattened parameters from FSDP state dicts.
    
    Args:
        model: An unwrapped model (ELECTRA or MixedClassifier)
    
    Returns:
        OrderedDict mapping parameter names to their shapes as tuples
    """
    param_shapes = OrderedDict()
    
    # Get all parameters
    for name, param in model.named_parameters():
        param_shapes[name] = tuple(param.shape)
    
    # Get all buffers (like running stats in BatchNorm)
    for name, buffer in model.named_buffers():
        param_shapes[name] = tuple(buffer.shape)
    
    return param_shapes


def print_peak_memory(accelerator: Accelerator):
    """Print peak memory usage across all ranks."""
    
    if not torch.cuda.is_available():
        return
    
    peak_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
    
    rank_data = {
        'rank': accelerator.process_index,
        'peak_gb': peak_gb
    }
    
    all_ranks = accelerator.gather_for_metrics([rank_data])
    
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("PEAK MEMORY USAGE")
        print("="*60)
        for data in all_ranks:
            print(f"Rank {data['rank']}: {data['peak_gb']:.3f} GB")
        print("="*60 + "\n")
    
    accelerator.wait_for_everyone()