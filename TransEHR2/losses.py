import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from TransEHR2.data.custom_types import EventAssociatedTensorData, MixedTensorDataset, ValueAssociatedTensorData
from TransEHR2.utils import calc_time_diff


class MaskedGeneratorLoss(torch.nn.Module):
    """
    Loss function for MaskedTokenGenerator that handles multiple output types:
    - Numeric features: Squared L2 norm of the difference between targets and predictions, averaged across samples.
    - Categorical features: Cross entropy loss with ignored padding class
    - Text features: Cosine similarity loss (rescaled from [-1,1] to [0,1])
    """
    
    def __init__(
        self, 
        numeric_weight: float = 1.0, 
        categorical_weight: float = 1.0, 
        text_weight: float = 1.0, 
        indicator_weight: float = 1.0
    ):
        """
        Initialize MultiOutputLoss with optional weights for different feature types.
        
        Args:
            numeric_weight (float): Weight for numeric feature loss
            categorical_weight (float): Weight for categorical feature loss
            text_weight (float): Weight for text feature loss
            indicator_weight (float): Weight for indicator prediction loss
        """
        super().__init__()
        self.numeric_weight = numeric_weight
        self.categorical_weight = categorical_weight
        self.text_weight = text_weight
        self.indicator_weight = indicator_weight
        
        # Define loss functions
        #
        # NOTE The feature-wise loss for numeric features is the squared L2 norm of the difference between targets and 
        # predictions. The sample-wise loss is the sum of the feature-wise losses across all features. Note that
        # this differs from MSELoss with reduction set to 'mean', which would rescale the feature-wise loss by the
        # number of dimensions in the feature vector. On the one hand, using 'mean' reduction would compensate for   
        # loss inflation from high-dimensional features when features dimensions vary widely. On the other
        # hand, the current implementation takes the straight-line distance between the target and prediction, 
        # treating all features as equal regardless of their dimensions. The current implementation requires that the
        # MSELoss reduction be set to 'none' followed by summation across feature components in the forward pass.
        self.mse_loss = torch.nn.MSELoss(reduction='none') 
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def _calculate_indicator_loss(self, pred_indicators, target_indicators, feature_masks):
        """
        Calculate indicator prediction loss for masked values.
        
        Args:
            pred_indicators (Tensor): Predicted indicators
            target_indicators (Tensor): Ground truth indicators
            feature_masks (Tensor): Mask indicating which features were masked
            
        Returns:
            Tuple[Tensor, int]: The loss value and number of masked elements
        """
        # Calculate BCE loss
        indicator_loss = self.bce_loss(pred_indicators, target_indicators.float())
        
        # Apply mask and sum
        masked_indicator_loss = (indicator_loss * feature_masks.bool()).sum()
        
        # Count number of masked elements
        n_masked = feature_masks.bool().sum().item()
        
        return masked_indicator_loss, n_masked
        
    def forward(
        self, 
        predictions: ValueAssociatedTensorData, 
        targets: MixedTensorDataset, 
        record_masks: Dict[str, Dict[str, Union[Tensor, List[Tensor]]]]
    ) -> Tensor:
        """
        Compute combined loss across all feature types.
        
        Args:
            predictions (ValueAssociatedTensorData): Output from MaskedTokenGenerator with predicted values
            targets (MixedTensorDataset): Original batch data with ground truth values
            record_masks (Dict): Dictionary of masks indicating which values were masked during generation
                
        Returns:
            Tensor: Scalar loss value combining all feature types
        """

        total_loss = 0.0
        n_masked = 0  # Count of masked values for normalization
        
        # Process numeric features
        if 'numeric' in predictions and 'numeric' in targets['val_data']:
            # Get predicted values
            pred_values = predictions['numeric']['values']  # List of tensors, one per feature
            # Get ground truth values
            target_values = targets['val_data']['numeric']['values']  # List of tensors, one per feature
            # Get masks for numeric features
            feature_masks = record_masks['numeric']['indicators']  # [batch_size, max_ts_len, n_numeric_feats]
            value_masks = record_masks['numeric']['values']  # List of [batch_size, max_ts_len, feature_dim]
            
            numeric_loss = 0.0
            
            # Process each numeric feature
            for f, (pred, target, value_mask) in enumerate(zip(pred_values, target_values, value_masks)):
                # Extract feature mask for this feature
                feat_mask = feature_masks[:, :, f].bool()  # [batch_size, max_ts_len]
                
                # Skip if no masked values
                if not feat_mask.any():
                    continue
                
                # Calculate loss only for masked positions
                feature_loss = self.mse_loss(pred, target)  # [batch_size, max_ts_len, feature_dim]

                # Apply the component-level mask (vectorized)
                mask = value_mask.bool()
                masked_loss = torch.where(mask, feature_loss, torch.zeros_like(feature_loss))

                # Reshape for vectorized operations
                # Create mask for valid entries (batch, time positions where feature is masked)
                batch_time_mask = feat_mask.unsqueeze(-1)  # [batch_size, max_ts_len, 1]
                
                # Sum across feature dimensions for each (batch, time) position, then mask with batch_time_mask
                # to get only the loss for positions where feature is masked
                pos_loss = masked_loss.sum(dim=-1, keepdim=True) # batch_size, max_ts_len, 1]
                pos_loss = torch.where(batch_time_mask, pos_loss, torch.zeros_like(pos_loss))

                # Sum total loss and count masked components
                numeric_loss += pos_loss.sum()
                
                # Count masked components (vectorized)
                # For each (batch, time) position where feature is masked, count number of component masks
                component_counts = (value_mask.bool() & batch_time_mask).sum()
                n_masked += component_counts.item()
            
            total_loss += self.numeric_weight * numeric_loss
                
            # Process indicator predictions if available
            if 'indicators' in predictions['numeric'] and predictions['numeric']['indicators'] is not None:
                pred_indicators = predictions['numeric']['indicators']  # [batch_size, max_ts_len, n_numeric_feats]
                target_indicators = targets['val_data']['numeric']['indicators']  # [batch_size, max_ts_len, n_numeric_feats]
                

                # Calculate indicator loss
                indicator_loss, indicator_n_masked = self._calculate_indicator_loss(
                    pred_indicators, target_indicators, feature_masks
                )
                
                total_loss += self.indicator_weight * indicator_loss
                n_masked += indicator_n_masked
        
        # Process categorical features
        if 'categorical' in predictions and 'categorical' in targets['val_data']:
            # Get predicted values
            pred_values = predictions['categorical']['values']  # List of tensors, one per feature
            
            # Get ground truth values - need to convert from one-hot to class indices
            target_values = []
            for target in targets['val_data']['categorical']['values']:
                # Convert one-hot encoding to class indices (vectorized)
                target_class = torch.argmax(target, dim=-1)
                target_values.append(target_class)
            
            # Get masks for categorical features
            feature_masks = record_masks['categorical']['indicators']  # [batch_size, max_ts_len, n_cat_feats]
            
            cat_loss = 0.0
            
            # Process each categorical feature
            for f, (pred, target) in enumerate(zip(pred_values, target_values)):
                # Extract feature mask for this feature
                feat_mask = feature_masks[:, :, f].bool().flatten()  # [batch_size * max_ts_len]
                
                # Skip if no masked values
                if not feat_mask.any():
                    continue
                
                # Reshape for cross entropy loss (vectorized)
                batch_size, max_ts_len, n_classes = pred.shape
                pred_flat = pred.reshape(-1, n_classes)
                target_flat = target.reshape(-1)
                
                # Calculate CE loss for all positions
                feature_loss = self.ce_loss(pred_flat, target_flat)  # [batch_size * max_ts_len]
                
                # Apply mask and sum
                masked_loss = feature_loss * feat_mask
                cat_loss += masked_loss.sum()
                
                # Count masked elements
                n_masked += feat_mask.sum().item()
            
            total_loss += self.categorical_weight * cat_loss
            
            # Process categorical indicators if available
            if 'indicators' in predictions['categorical'] and predictions['categorical']['indicators'] is not None:
                pred_indicators = predictions['categorical']['indicators']
                target_indicators = targets['val_data']['categorical']['indicators']
                
                # Calculate indicator loss
                indicator_loss, indicator_n_masked = self._calculate_indicator_loss(
                    pred_indicators, target_indicators, feature_masks
                )
                
                total_loss += self.indicator_weight * indicator_loss
                n_masked += indicator_n_masked
        
        # Process text features
        if 'text' in predictions and 'text' in targets['val_data']:
            # Get predicted text embeddings
            pred_values = predictions['text']['values']  # List of tensors, one per feature
            
            # Get LLM-generated embeddings from target tokens
            # The LLM embeddings should already be computed in the targets
            if 'embedded_values' in targets['val_data']['text']:
                target_values = targets['val_data']['text']['embedded_values']
                has_target_embeddings = True
            else:
                target_values = None
                has_target_embeddings = False
            
            if not has_target_embeddings:
                # If no pre-computed embeddings, we can't calculate loss for text
                pass
            else:
                # Get masks for text features
                feature_masks = record_masks['text']['indicators']  # [batch_size, max_ts_len, n_text_feats]
                value_masks = record_masks['text']['embedded_values']  # List of [batch_size, max_ts_len, TEXT_EMBED_DIM]
                
                text_loss = 0.0
                
                # Process each text feature
                for f, (pred, target, value_mask) in enumerate(zip(pred_values, target_values, value_masks)):
                    # Extract feature mask for this feature
                    feat_mask = feature_masks[:, :, f].bool()  # [batch_size, max_ts_len]
                    
                    # Skip if no masked values
                    if not feat_mask.any():
                        continue
                    
                    # Normalize embeddings for cosine similarity (vectorized)
                    pred_norm = F.normalize(pred, p=2, dim=-1)
                    target_norm = F.normalize(target, p=2, dim=-1)
                    
                    # Calculate cosine similarity (dot product of normalized vectors)
                    # This is a vectorized operation over the embedding dimension
                    cosine_sim = torch.sum(pred_norm * target_norm, dim=-1)  # [batch_size, max_ts_len]
                    
                    # Convert similarity [-1,1] to distance [0,2] and rescale to [0,1]
                    cosine_distance = (1.0 - cosine_sim) / 2.0
                    
                    # Apply feature mask
                    masked_distance = cosine_distance * feat_mask
                    
                    # Calculate masked components (vectorized)
                    # Create 3D mask combining feature mask and component mask
                    combined_mask = feat_mask.unsqueeze(-1) & value_mask.bool()  # [batch_size, max_ts_len, TEXT_EMBED_DIM]
                    
                    # Count masked components
                    component_count = combined_mask.sum().item()
                    
                    # If using component-level masking
                    if component_count > 0:
                        # Adjust the cosine distance to account for partial embedding masking
                        # For text, we count each masked embedding as one loss unit regardless of components
                        # because semantic meaning spans the entire embedding
                        text_loss += masked_distance.sum()
                        n_masked += feat_mask.sum().item()
                    else:
                        # Just use feature-level mask
                        text_loss += masked_distance.sum()
                        n_masked += feat_mask.sum().item()
                
                total_loss += self.text_weight * text_loss
                
                # Process text indicators if available
                if 'indicators' in predictions['text'] and predictions['text']['indicators'] is not None:
                    pred_indicators = predictions['text']['indicators']
                    target_indicators = targets['val_data']['text']['indicators']
                    
                    # Calculate indicator loss
                    indicator_loss, indicator_n_masked = self._calculate_indicator_loss(
                        pred_indicators, target_indicators, feature_masks
                    )
                    
                    total_loss += self.indicator_weight * indicator_loss
                    n_masked += indicator_n_masked
        
        # Normalize the total loss by the number of masked values
        if n_masked > 0:
            total_loss = total_loss / n_masked
        
        return total_loss


class MaskedDiscriminatorLoss(torch.nn.Module):
    """
    Loss function for MaskedTokenDiscriminator that predicts whether each feature
    at each timestep was real or generated by the MaskedTokenGenerator.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize MaskedDiscriminatorLoss.
        
        Args:
            weight (float): Weight for the discriminator loss in combined training. Defaults to 1.0.
        """
        super().__init__()
        self.weight = weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self, 
        predictions: Dict[str, Tensor], 
        record_masks: Dict[str, Dict[str, Union[Tensor, List[Tensor]]]]
    ) -> Tensor:
        """
        Compute discriminator loss for distinguishing real vs generated features.
        
        Args:
            predictions (Dict[str, Tensor]): Output from MaskedTokenDiscriminator with predictions for each feature type.
                Expected structure:
                {
                    'numeric': Tensor of shape (batch_size, max_ts_len, n_numeric_feats),
                    'categorical': Tensor of shape (batch_size, max_ts_len, n_categorical_feats),
                    'text': Tensor of shape (batch_size, max_ts_len, n_text_feats)
                }
            record_masks (Dict): Dictionary of masks indicating which values were masked during generation.
                A value of 1 indicates a record was masked (and should be predicted as generated),
                0 indicates it was not masked (and should be predicted as real).
                
        Returns:
            Tensor: Scalar loss value for the discriminator
        """
        total_loss = 0.0
        n_predictions = 0
        
        # Process each feature type
        for feat_type in ['numeric', 'categorical', 'text']:
            if feat_type not in predictions or feat_type not in record_masks:
                continue
                
            pred_logits = predictions[feat_type]  # (batch_size, max_ts_len, n_features)
            indicator_mask = record_masks[feat_type]['indicators']  # (batch_size, max_ts_len, n_features)
            
            # Create target labels: 1 for generated (masked), 0 for real (not masked)
            targets = indicator_mask.float()
            
            # Calculate BCE loss for all positions
            feature_loss = self.bce_loss(pred_logits, targets)  # (batch_size, max_ts_len, n_features)
            
            # Sum loss across all dimensions
            total_loss += feature_loss.sum()
            n_predictions += feature_loss.numel()
        
        # Normalize by number of predictions
        if n_predictions > 0:
            total_loss = total_loss / n_predictions
        
        return self.weight * total_loss


class TransformerHawkesLoss(torch.nn.Module):
    """
    Loss function for TransformerHawkesProcess based on log-likelihood of event sequences.
    Combines event log-likelihood and non-event log-likelihood using Monte Carlo integration.
    """
    
    def __init__(
        self,
        add_prediction_loss: bool = True,
        nll_weight: float = 1.0,
        type_weight: float = 1.0,
        time_weight: float = 0.01
    ):
        """
        Initialize TransformerHawkesLoss.
        
        Args:
            add_prediction_loss (bool): If True (default), the cross-entropy loss for event type prediction and scaled 
                MSE loss for the event time prediction are added to the Hawkes process log-likelihood loss.
            nll_weight (float): Weight applied to the negative log-likelihood. Defaults to 1.0.
            type_weight (float): Weight applied to the event type prediction loss if add_prediction_loss is True. 
                Defaults to 1.0.
            time_weight (float): Weight applied to the time prediction loss if add_prediction_loss is True. Defaults to 
                0.01.
        """
        super().__init__()
        self.add_prediction_loss = add_prediction_loss
        self.nll_weight = nll_weight

        if add_prediction_loss:
            # BCE loss for multi-label event type prediction (expects logits)
            # MSE loss for event time prediction will be scaled by time_weight
            self.type_weight = type_weight
            self.time_weight = time_weight
            self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.mse_loss = torch.nn.MSELoss(reduction='none')

    def _log_likelihood(
        self,
        observed_initial_intensities: Tensor,
        observed_conditional_intensities: Tensor,
        sampled_intensities: Tensor,
        event_data: EventAssociatedTensorData
    ) -> Tuple[Tensor, Tensor]:
        """Log-likelihood of sequence.

        The estimated values of the intensity functions at each time step are based on equation (6) in Zuo et al., but with time intervals corrected to conform to the accepted general definition of a Hawkes process. The interpolation is done over left-continuous time intervals to ensure that the log-likelihood is correctly calculated, whereas Zuo et al. defined the interpolation over right-continuous intervals. Whereas Zuo et al. did not define the likelihood for the first time step in the sequence, this implementation defines the intensity at the first time step based on the model's base intensity parameters alone.

        Args:
            observed_initial_intensities (Tensor): Intensity function values for the first observed event in the 
                sequence. The tensor should have a shape like (batch_size, 1, n_event_types).
            observed_conditional_intensities (Tensor): Intensity function values for observed events after the first 
                one. The tensor should have a shape like (batch_size, max_ts_len - 1, n_event_types).
            sampled_intensities (Tensor): Intensity function values for inter-event time samples of shape 
                (batch_size, max_ts_len - 1, n_samples, n_event_types)
            event_data (EventAssociatedTensorData): Event data containing:
                - 'indicators': Tensor of shape (batch_size, max_ts_len, n_event_types) with event type indicator bits
                - 'times': Tensor of shape (batch_size, max_ts_len) with event timestamps
                - 'masks': Tensor of shape (batch_size, max_ts_len) with non-padding masks (0 for padding, 1 otherwise)
    
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - event_ll (Tensor): Log-likelihood of observed events of shape (batch_size, max_ts_len)
                - non_event_ll (Tensor): Log-likelihood of non-events of shape (batch_size, max_ts_len)
        """
        eps = torch.finfo(torch.float32).eps  # Small constant to avoid log(0)

        event_indicators = event_data['indicators']  # (batch_size, max_ts_len, n_event_types)
        event_times = event_data['times']  # (batch_size, max_ts_len)
        non_pad_mask = event_data['masks']  # (batch_size, max_ts_len)

        # Difference between consecutive timestamps for interpolating the intensity function
        #     Shape: (batch_size, max_ts_len)
        time_diff_obs = calc_time_diff(event_times, non_pad_mask, device=event_times.device)  # First column is zeros
        
        # Compute event log-likelihood  
        observed_intensities = torch.cat([observed_initial_intensities, observed_conditional_intensities], dim=1)
        obs_log_intensities = torch.log(observed_intensities + eps)  # (batch_size, max_ts_len, n_event_types)
        event_ll = torch.sum(obs_log_intensities * event_indicators * non_pad_mask[..., None], dim=-1)
        
        # Compute the non-event log-likelihood for the first event based on the initial intensity
        #    Shape: (batch_size, 1)
        initial_non_event_ll = observed_initial_intensities.sum(dim=-1) * non_pad_mask[:, [0]]  # Sum over event types
        
        # Compute the non-event log-likelihood with Monte Carlo integration for events after the first one
        #     sampled_intensity_totals shape: (batch_size, max_ts_len - 1, n_samples)
        #     cond_non_event_ll shape: (batch_size, max_ts_len - 1)
        sampled_intensity_totals = sampled_intensities.sum(dim=-1)  # Sum over event types
        cond_non_event_ll = sampled_intensity_totals.mean(dim=-1) * time_diff_obs[:, 1:] * non_pad_mask[:, 1:]
        
        # Concatenate the initial and conditional non-event log-likelihoods
        non_event_ll = torch.cat([initial_non_event_ll, cond_non_event_ll], dim=1)

        return event_ll, non_event_ll
    
    def forward(
        self,
        observed_initial_intensities: Tensor,
        observed_conditional_intensities: Tensor,
        sampled_intensities: Tensor,
        event_data: EventAssociatedTensorData,
        type_preds: Optional[Tensor] = None,
        time_preds: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute negative log-likelihood loss for event sequences.

        Calculates the negative log-likelihood of the model given the observed event sequences. If self.add_prediction_loss is True, the forward pass also adds the cross entropy loss for multi-label predictions of the event type(s) and the scaled MSE loss for the predicted time delta to the next event(s) as described in the results section of Zuo et al. (https://arxiv.org/pdf/2002.09291). Negative log-likelihood loss is averaged across all batch items. The losses for event type and time, if used, are averaged across all batch items and non-padding time steps.
        
        Args:
            observed_initial_intensities (Tensor): Intensity function values for the first observed event in the 
                sequence. The tensor should have a shape like (batch_size, 1, n_event_types).
            observed_conditional_intensities (Tensor): Intensity function values for observed events after the first 
                one. The tensor should have a shape like (batch_size, max_ts_len - 1, n_event_types).
            sampled_intensities (Tensor): Intensity function values for inter-event time samples of shape 
                (batch_size, max_ts_len - 1, n_samples, n_event_types)
            event_data (EventAssociatedTensorData): Dictionary containing:
                - 'indicators': Event indicators of shape (batch_size, max_ts_len, n_event_types)
                - 'times': Event timestamps of shape (batch_size, max_ts_len)
                - 'masks': Non-padding mask of shape (batch_size, max_ts_len)
            type_preds (Optional[Tensor]): Predicted event types of shape (batch_size, max_ts_len, n_event_types)
            time_preds (Optional[Tensor]): Predicted event times of shape (batch_size, max_ts_len, 1)
                
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]: A tuple containing the overall loss used for optimization 
                and a tuple of unweighted NLL, type, and time losses, respectively, which can be examined for analysis. If self.add_prediction_loss is False, the component losses for type and time predictions will be zero tensors.
        """
        # Unpack event data
        event_indicators = event_data['indicators'] # (batch_size, max_ts_len, n_event_types)
        event_times = event_data['times']  # (batch_size, max_ts_len)
        non_pad_mask = event_data['masks']  # (batch_size, max_ts_len)

        # Calculate event and non-event log-likelihoods using existing utility functions
        event_ll, non_event_ll = self._log_likelihood(
            observed_initial_intensities, 
            observed_conditional_intensities, 
            sampled_intensities, 
            event_data
        )

        # Return negative log-likelihood as loss (we want to maximize likelihood)
        # Sum over time steps, then take the mean over batch items
        nll_loss = torch.mean(-torch.sum(event_ll - non_event_ll, dim=-1))

        nll_weight = torch.tensor(self.nll_weight, device=nll_loss.device, dtype=nll_loss.dtype)
        type_weight = torch.tensor(self.type_weight, device=nll_loss.device, dtype=nll_loss.dtype)
        time_weight = torch.tensor(self.time_weight, device=nll_loss.device, dtype=nll_loss.dtype)

        if self.add_prediction_loss:
            # Predictions at the final time step are ignored because there is no target
            if type_preds is None or time_preds is None:
                raise ValueError("type_preds and time_preds must be provided when add_prediction_loss is True.")
            
            # Calculate BCE loss for event types and mask out padding time steps
            #    shape: (batch_size, max_ts_len - 1, n_event_types)
            type_loss = self.bce_loss(type_preds[:, :-1, :], event_indicators[:, 1:, :].float())
            # Apply non-padding mask and sum over all dimensions
            type_loss = (type_loss * non_pad_mask[:, 1:, None]).sum()  # Scalar
            # Calculate the mean across all batch items, non-padding time steps, and event types
            n_type_preds = (non_pad_mask[:, 1:, None] * torch.ones_like(event_indicators[:, 1:, :])).sum().item()
            type_loss = type_loss / n_type_preds if n_type_preds > 0 else type_loss

            # Calculate MSE loss for time predictions
            #     shape: (batch_size, max_ts_len - 1)
            time_preds = time_preds.squeeze(-1)[:, :-1]
            time_targets = calc_time_diff(event_times, non_pad_mask, device=event_times.device)[:, 1:]
            time_loss = self.mse_loss(time_preds, time_targets) * non_pad_mask[:, 1:]
            # Calculate the mean across all batch items and non-padding time steps
            n_obs = non_pad_mask[:, 1:].sum().item()
            time_loss = time_loss.sum() / n_obs if n_obs > 0 else time_loss.sum()

            # Combine losses
            overall_loss = (nll_weight * nll_loss) + (type_weight * type_loss) + (time_weight * time_loss)

        else:
            type_loss = torch.tensor(0.0, device=nll_loss.device)
            time_loss = torch.tensor(0.0, device=nll_loss.device)
            overall_loss = nll_weight * nll_loss

        return overall_loss, (nll_loss, type_loss, time_loss)
    