import torch
import torch.nn.functional as F

from dlordinal.losses import BetaLoss
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from TransEHR2.data.custom_types import EventAssociatedTensorData, MixedTensorDataset, ValueAssociatedTensorData
from TransEHR2.utils import calc_time_diff


# Type alias for the sparse masked targets structure returned by ELECTRA._extract_masked_targets()
MaskedTargets = Dict[str, Dict[str, Union[List[Tensor], Tensor]]]


class MaskedGeneratorLoss(torch.nn.Module):
    """
    Loss function for MaskedTokenGenerator that handles multiple output types:
    - Numeric features: Squared L2 norm of the difference between targets and predictions, averaged across samples.
    - Categorical features: Cross entropy loss with ignored padding class
    - Text features: Cosine similarity loss (rescaled from [-1,1] to [0,1])
    
    This loss function supports two input formats for targets:
    1. Full batch format (MixedTensorDataset): The original format with complete batch data
    2. Sparse masked targets format (MaskedTargets): Memory-efficient format containing only
       values at masked positions, as returned by ELECTRA._extract_masked_targets()
    """
    
    def __init__(
        self,
        numeric_weight: float = 1.0,
        categorical_weight: float = 1.0,
        ordinal_weight: float = 1.0,
        text_weight: float = 1.0,
        indicator_weight: float = 1.0,
        ordinal_features: Optional[List[int]] = None
    ):
        """
        Initialize MultiOutputLoss with optional weights for different feature types.

        Args:
            numeric_weight (float): Weight for numeric feature loss
            categorical_weight (float): Weight for categorical feature loss
            ordinal_weight (float): Weight for ordinal feature loss
            text_weight (float): Weight for text feature loss
            indicator_weight (float): Weight for indicator prediction loss
            ordinal_features (List[int], optional): List of number of levels for
                each ordinal feature. Required if ordinal features are present.
        """
        super().__init__()
        self.numeric_weight = numeric_weight
        self.categorical_weight = categorical_weight
        self.ordinal_weight = ordinal_weight
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

        # Ordinal loss: BetaLoss wrapping CrossEntropyLoss, one per ordinal feature
        if ordinal_features:
            self.ordinal_beta_losses = torch.nn.ModuleList([
                BetaLoss(
                    base_loss=torch.nn.CrossEntropyLoss(reduction='none'),
                    num_classes=n_levels
                )
                for n_levels in ordinal_features
            ])
        else:
            self.ordinal_beta_losses = None
    
    def _calculate_indicator_loss_sparse(
        self, 
        pred_indicators: Tensor, 
        target_indicators: Tensor,
        indicator_mask: Tensor
    ) -> Tuple[Tensor, int]:
        """
        Calculate indicator prediction loss using sparse masked targets.
        
        Args:
            pred_indicators: Predicted indicators, shape (batch_size, max_ts_len, n_features)
            target_indicators: Ground truth indicators at masked positions only (1D flattened tensor)
            indicator_mask: Boolean mask indicating which positions were masked
            
        Returns:
            Tuple[Tensor, int]: The loss value and number of masked elements
        """
        # Extract predictions at masked positions to match sparse target format
        pred_at_masked = pred_indicators[indicator_mask]
        # Cast target to match prediction dtype for mixed precision compatibility
        target_casted = target_indicators.to(pred_at_masked.dtype)
        # Calculate BCE loss on the sparse tensors
        indicator_loss = self.bce_loss(pred_at_masked, target_casted)
        # Sum the loss
        masked_indicator_loss = indicator_loss.sum()
        n_masked = target_indicators.numel()
        
        return masked_indicator_loss, n_masked
        
    def _forward_sparse(
        self,
        predictions: ValueAssociatedTensorData,
        masked_targets: MaskedTargets,
        record_masks: Dict[str, Dict[str, Union[Tensor, List[Tensor]]]]
    ) -> Tensor:
        """
        Compute loss using sparse masked targets format (memory-efficient).
        
        This method computes loss using only the pre-extracted masked values, avoiding the need
        to access the full batch tensors. This is more memory-efficient when the batch has already
        been modified in-place for the discriminator.
        
        Args:
            predictions: Output from MaskedTokenGenerator with predicted values
            masked_targets: Sparse structure containing only values at masked positions
            record_masks: Dictionary of masks indicating which values were masked
                
        Returns:
            Tensor: Scalar loss value combining all feature types
        """
        total_loss = 0.0
        n_masked = 0
        
        # Process numeric features
        if 'numeric' in predictions and 'numeric' in masked_targets:
            pred_values = predictions['numeric']['values']
            target_values = masked_targets['numeric']['values']
            feature_masks = record_masks['numeric']['indicators']
            value_masks = record_masks['numeric']['values']
            
            numeric_loss = 0.0
            numeric_n_masked = 0
            for f, (pred, target) in enumerate(zip(pred_values, target_values)):
                if target.numel() == 0:
                    continue
                value_mask = value_masks[f].bool()
                pred_at_masked = pred[value_mask]
                target_casted = target.to(pred_at_masked.dtype)
                feature_loss = self.mse_loss(pred_at_masked, target_casted)
                numeric_loss += feature_loss.sum()
                numeric_n_masked += target.numel()
            
            total_loss += self.numeric_weight * numeric_loss
            n_masked += numeric_n_masked
            
            # Process indicator predictions if available
            if 'indicators' in predictions['numeric'] and predictions['numeric']['indicators'] is not None:
                if 'indicators' in masked_targets['numeric']:
                    indicator_loss, indicator_n_masked = self._calculate_indicator_loss_sparse(
                        predictions['numeric']['indicators'],
                        masked_targets['numeric']['indicators'],
                        feature_masks.bool()
                    )
                    total_loss += self.indicator_weight * indicator_loss
                    n_masked += indicator_n_masked
        
        # Process categorical features
        if 'categorical' in predictions and 'categorical' in masked_targets:
            pred_values = predictions['categorical']['values']
            target_values = masked_targets['categorical']['values']
            feature_masks = record_masks['categorical']['indicators']
            
            cat_loss = 0.0
            cat_n_masked = 0
            for f, (pred, target) in enumerate(zip(pred_values, target_values)):
                if target.numel() == 0:
                    continue
                feat_mask = feature_masks[:, :, f].bool()
                pred_at_masked = pred[feat_mask]
                target_classes = torch.argmax(target, dim=-1)
                feature_loss = self.ce_loss(pred_at_masked, target_classes)
                cat_loss += feature_loss.sum()
                cat_n_masked += target_classes.numel()
            
            total_loss += self.categorical_weight * cat_loss
            n_masked += cat_n_masked
            
            # Process categorical indicators
            if 'indicators' in predictions['categorical'] and predictions['categorical']['indicators'] is not None:
                if 'indicators' in masked_targets['categorical']:
                    indicator_loss, indicator_n_masked = self._calculate_indicator_loss_sparse(
                        predictions['categorical']['indicators'],
                        masked_targets['categorical']['indicators'],
                        feature_masks.bool()
                    )
                    total_loss += self.indicator_weight * indicator_loss
                    n_masked += indicator_n_masked
        
        # Process ordinal features
        if 'ordinal' in predictions and 'ordinal' in masked_targets and self.ordinal_beta_losses is not None:
            pred_values = predictions['ordinal']['values']
            target_values = masked_targets['ordinal']['values']
            feature_masks = record_masks['ordinal']['indicators']

            ord_loss = 0.0
            ord_n_masked = 0
            for f, (pred, target, beta_loss) in enumerate(
                zip(pred_values, target_values, self.ordinal_beta_losses)
            ):
                if target.numel() == 0:
                    continue
                feat_mask = feature_masks[:, :, f].bool()
                # pred shape: (batch, max_ts, n_levels) — CLM probabilities
                pred_at_masked = pred[feat_mask]  # (n_masked, n_levels)
                # target shape: (n_masked, 1) — 1-indexed class indices
                # Convert to 0-indexed for BetaLoss
                target_classes = (target[:, 0] - 1).long()
                # BetaLoss expects (N, J) probs and (N,) targets
                feature_loss = beta_loss(pred_at_masked, target_classes)
                ord_loss += feature_loss.sum()
                ord_n_masked += target_classes.numel()

            total_loss += self.ordinal_weight * ord_loss
            n_masked += ord_n_masked

            # Process ordinal indicators
            if 'indicators' in predictions['ordinal'] and predictions['ordinal']['indicators'] is not None:
                if 'indicators' in masked_targets['ordinal']:
                    indicator_loss, indicator_n_masked = self._calculate_indicator_loss_sparse(
                        predictions['ordinal']['indicators'],
                        masked_targets['ordinal']['indicators'],
                        feature_masks.bool()
                    )
                    total_loss += self.indicator_weight * indicator_loss
                    n_masked += indicator_n_masked

        # Process text features
        if 'text' in predictions and 'text' in masked_targets:
            if 'embedded_values' in masked_targets['text']:
                pred_values = predictions['text']['embedded_values']
                target_values = masked_targets['text']['embedded_values']
                feature_masks = record_masks['text']['indicators']

                text_loss = 0.0
                text_n_masked = 0
                for f, (pred, target) in enumerate(zip(pred_values, target_values)):
                    if target.numel() == 0:
                        continue
                    feat_mask = feature_masks[:, :, f].bool()
                    pred_at_masked = pred[feat_mask]
                    target_casted = target.to(pred_at_masked.dtype)
                    # Filter out zero vectors to avoid NaN from F.normalize
                    pred_norms = torch.norm(pred_at_masked, p=2, dim=-1)
                    target_norms = torch.norm(target_casted, p=2, dim=-1)
                    # Create valid mask: feature is masked AND both vectors are non-zero
                    eps = 1e-8
                    valid_mask = (pred_norms > eps) & (target_norms > eps)
                    if valid_mask.sum() == 0:
                        continue
                    # Extract valid vectors
                    pred_valid = pred_at_masked[valid_mask]
                    target_valid = target_casted[valid_mask]
                    # Normalize embeddings for cosine similarity
                    pred_norm = F.normalize(pred_valid, p=2, dim=-1)
                    target_norm = F.normalize(target_valid, p=2, dim=-1)
                    # Calculate cosine similarity (dot product of normalized vectors)
                    cosine_sim = torch.sum(pred_norm * target_norm, dim=-1)
                    # Convert similarity [-1,1] to distance [0,2] and rescale to [0,1]
                    cosine_distance = (1.0 - cosine_sim) / 2.0

                    # For text, count each valid masked embedding as one loss unit
                    text_loss += cosine_distance.sum()
                    text_n_masked += cosine_distance.numel()

                total_loss += self.text_weight * text_loss
                n_masked += text_n_masked

                # Process text indicators
                if 'indicators' in predictions['text'] and predictions['text']['indicators'] is not None:
                    if 'indicators' in masked_targets['text']:
                        indicator_loss, indicator_n_masked = self._calculate_indicator_loss_sparse(
                            predictions['text']['indicators'],
                            masked_targets['text']['indicators'],
                            feature_masks.bool()
                        )
                        total_loss += self.indicator_weight * indicator_loss
                        n_masked += indicator_n_masked

        # Normalize the total loss by the number of masked values
        if n_masked > 0:
            total_loss = total_loss / n_masked

        return total_loss
    
    def _forward_full_batch(
        self,
        predictions: ValueAssociatedTensorData,
        targets: MixedTensorDataset,
        record_masks: Dict[str, Dict[str, Union[Tensor, List[Tensor]]]]
    ) -> Tensor:
        """
        Compute loss using full batch format (original implementation).
        
        This is the original loss computation that expects the complete batch with
        targets['val_data'] containing all values.
        
        Args:
            predictions: Output from MaskedTokenGenerator with predicted values
            targets: Full batch data with ground truth values in targets['val_data']
            record_masks: Dictionary of masks indicating which values were masked
                
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
            
            # Process each numeric feature
            numeric_loss = 0.0
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
                pos_loss = masked_loss.sum(dim=-1, keepdim=True)  # [batch_size, max_ts_len, 1]
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
                target_indicators = targets['val_data']['numeric']['indicators']  # [b_sz, max_ts_len, n_numeric_fts]
                # Calculate indicator loss
                indicator_loss = self.bce_loss(pred_indicators, target_indicators.float())
                masked_indicator_loss = (indicator_loss * feature_masks.bool()).sum()
                indicator_n_masked = feature_masks.bool().sum().item()
                
                total_loss += self.indicator_weight * masked_indicator_loss
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
                indicator_loss = self.bce_loss(pred_indicators, target_indicators.float())
                masked_indicator_loss = (indicator_loss * feature_masks.bool()).sum()
                indicator_n_masked = feature_masks.bool().sum().item()

                total_loss += self.indicator_weight * masked_indicator_loss
                n_masked += indicator_n_masked
        
        # Process ordinal features
        if 'ordinal' in predictions and 'ordinal' in targets['val_data'] and self.ordinal_beta_losses is not None:
            pred_values = predictions['ordinal']['values']  # List of (batch, max_ts, n_levels)
            # Get masks for ordinal features
            feature_masks = record_masks['ordinal']['indicators']  # [batch_size, max_ts_len, n_ord_feats]

            ord_loss = 0.0
            for f, (pred, beta_loss) in enumerate(
                zip(pred_values, self.ordinal_beta_losses)
            ):
                feat_mask = feature_masks[:, :, f].bool().flatten()  # [batch_size * max_ts_len]
                if not feat_mask.any():
                    continue
                batch_size, max_ts_len, n_levels = pred.shape
                pred_flat = pred.reshape(-1, n_levels)  # (B*T, n_levels)
                # Target: 1-indexed class index stored in (batch, max_ts, 1)
                target = targets['val_data']['ordinal']['values'][f]
                target_flat = (target[:, :, 0] - 1).long().reshape(-1)  # 0-indexed, (B*T,)
                # BetaLoss expects (N, J) probs and (N,) targets
                feature_loss = beta_loss(pred_flat, target_flat)  # (B*T,)
                masked_loss = feature_loss * feat_mask
                ord_loss += masked_loss.sum()
                n_masked += feat_mask.sum().item()

            total_loss += self.ordinal_weight * ord_loss

            # Process ordinal indicators if available
            if 'indicators' in predictions['ordinal'] and predictions['ordinal']['indicators'] is not None:
                pred_indicators = predictions['ordinal']['indicators']
                target_indicators = targets['val_data']['ordinal']['indicators']
                indicator_loss = self.bce_loss(pred_indicators, target_indicators.float())
                masked_indicator_loss = (indicator_loss * feature_masks.bool()).sum()
                indicator_n_masked = feature_masks.bool().sum().item()
                total_loss += self.indicator_weight * masked_indicator_loss
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
            
            if has_target_embeddings:
                # Process each text feature
                feature_masks = record_masks['text']['indicators']  # [batch_size, max_ts_len, n_text_feats]
                value_masks = record_masks['text']['embedded_values']  # List: [batch_size, max_ts_len, TEXT_EMBED_DIM]
                text_loss = 0.0
                for f, (pred, target, value_mask) in enumerate(zip(pred_values, target_values, value_masks)):
                    # Extract feature mask for this feature
                    feat_mask = feature_masks[:, :, f].bool()  # [batch_size, max_ts_len]
                    # Skip if no masked values
                    if not feat_mask.any():
                        continue
                    # Compute norms to identify zero vectors
                    pred_norms = torch.norm(pred, p=2, dim=-1)  # [batch_size, max_ts_len]
                    target_norms = torch.norm(target, p=2, dim=-1)  # [batch_size, max_ts_len]
                    # Create valid mask: feature is masked AND both vectors are non-zero
                    eps = 1e-8
                    valid_mask = feat_mask & (pred_norms > eps) & (target_norms > eps)
                    if not valid_mask.any():
                        continue
                    # Extract valid vectors
                    pred_valid = pred[valid_mask]  # [n_valid, TEXT_EMBED_DIM]
                    target_valid = target[valid_mask]  # [n_valid, TEXT_EMBED_DIM]
                    
                    # Normalize embeddings for cosine similarity
                    pred_norm = F.normalize(pred_valid, p=2, dim=-1)
                    target_norm = F.normalize(target_valid, p=2, dim=-1)
                    # Calculate cosine similarity (dot product of normalized vectors)
                    cosine_sim = torch.sum(pred_norm * target_norm, dim=-1)  # [n_valid]
                    # Convert similarity [-1,1] to distance [0,2] and rescale to [0,1]
                    cosine_distance = (1.0 - cosine_sim) / 2.0
                    
                    # For text, count each valid masked embedding as one loss unit
                    text_loss += cosine_distance.sum()
                    n_masked += cosine_distance.numel()
                
                total_loss += self.text_weight * text_loss
                
                # Process text indicators if available
                if 'indicators' in predictions['text'] and predictions['text']['indicators'] is not None:
                    if 'indicators' in targets['text']:
                        indicator_loss, indicator_n_masked = self._calculate_indicator_loss_sparse(
                            predictions['text']['indicators'],
                            targets['text']['indicators'],
                            feature_masks.bool()
                        )
                        total_loss += self.indicator_weight * indicator_loss
                        n_masked += indicator_n_masked
        
        # Normalize the total loss by the number of masked values
        if n_masked > 0:
            total_loss = total_loss / n_masked
        
        return total_loss
    

    def forward(
        self, 
        predictions: ValueAssociatedTensorData, 
        targets: Union[MixedTensorDataset, MaskedTargets],
        record_masks: Dict[str, Dict[str, Union[Tensor, List[Tensor]]]]
    ) -> Tensor:
        """
        Compute combined loss across all feature types.
        
        This method supports two target formats:
        1. Full batch (MixedTensorDataset): Contains complete batch with targets['val_data']
        2. Sparse masked targets (MaskedTargets): Contains only values at masked positions,
           as returned by ELECTRA._extract_masked_targets(). This format is more memory-efficient.
        
        The format is auto-detected based on whether 'val_data' key exists in targets.
        
        Args:
            predictions (ValueAssociatedTensorData): Output from MaskedTokenGenerator with predicted values
            targets: Either full batch data (MixedTensorDataset) or sparse masked targets (MaskedTargets)
            record_masks (Dict): Dictionary of masks indicating which values were masked during generation
                
        Returns:
            Tensor: Scalar loss value combining all feature types
        """
        # Auto-detect target format: sparse if no 'val_data' key, full batch otherwise
        use_sparse_targets = 'val_data' not in targets
        
        if use_sparse_targets:
            return self._forward_sparse(predictions, targets, record_masks)
        else:
            return self._forward_full_batch(predictions, targets, record_masks)
    


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
        for feat_type in ['numeric', 'categorical', 'ordinal', 'text']:
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
        eps = 1e-6 # Small constant to avoid log(0)

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
    