import torch

from copy import deepcopy
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from TransEHR2.constants import TEXT_EMBED_DIM
from TransEHR2.data.custom_types import MixedTensorDataset, ValueAssociatedTensorData
from TransEHR2.modules import MaskedTokenDiscriminator, MaskedTokenGenerator, TransformerHawkesProcess
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder
from TransEHR2.modules import GradientTraceableLLM
from TransEHR2.utils import calc_time_diff, sample_non_event_time_diff


class ELECTRA(torch.nn.Module):
    def __init__(
        self, 
        generator: MaskedTokenGenerator,
        discriminator: MaskedTokenDiscriminator,
        hawkes: TransformerHawkesProcess,
        use_text: bool = False,
        llm_module: Optional[GradientTraceableLLM] = None
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.hawkes = hawkes
        self.use_text = use_text
        
        # Use shared LLM if use_text and llm_module provided
        if self.use_text:
            if llm_module is not None:
                self.llm_module = llm_module
            else:
                self.llm_module = GradientTraceableLLM()
        else:
            self.llm_module = None

    def _gen_text_embeddings(
        self,
        value_data: ValueAssociatedTensorData,
        trace_grads: bool = False
    ) -> torch.Tensor:
        """
        Process text data through LLM, handling sparse text efficiently.
        
        Args:
            value_data: Dictionary containing text data
            trace_grads: Whether to trace gradients through LLM
            
        Returns:
            torch.Tensor: Text embeddings with shape [batch, timesteps, features, embed_dim]
        """
        # Stack the text tokens and masks
        text_tokens = torch.stack(value_data['text']['values'], dim=2)  # [batch, timesteps, features, tokens]
        text_token_masks = torch.stack(value_data['text']['masks'], dim=2)
        
        batch_size, max_ts_len, n_text_feats, max_tokens = text_tokens.shape
        
        # Create a mask to identify timesteps with actual text data
        has_text_per_timestep = (text_tokens.max(dim=-1)[0] > 0)  # [batch, timesteps, features]
        has_any_text = has_text_per_timestep.any()

        # Initialize embeddings tensor with zeros
        final_embeddings = torch.zeros(
            batch_size, max_ts_len, n_text_feats, TEXT_EMBED_DIM,
            device=text_tokens.device, 
            dtype=torch.float32
        )
        
        # If text is present, perform a forward pass only on those positions. If text is not present anywhere in the
        # batch, perform a forward pass with a dummy tensor to avoid NCCL timeouts due to desynchronization when using
        # multiple GPUs in distributed computation.

        if has_any_text:

            # Find positions where we have actual text
            text_positions = torch.where(has_text_per_timestep)
            
            # Extract only the tokens that have actual text
            batch_indices, timestep_indices, feature_indices = text_positions
            
            # Get the actual text tokens and masks for processing
            tokens_to_process = text_tokens[batch_indices, timestep_indices, feature_indices]  # [N, tokens]
            masks_to_process = text_token_masks[batch_indices, timestep_indices, feature_indices]  # [N, tokens]

            # Process through LLM
            #   llm_embeddings shape: [N, TEXT_EMBED_DIM]
            llm_embeddings = self.llm_module(
                token_ids=tokens_to_process, 
                trace_grads=trace_grads, 
                attention_mask=masks_to_process
            )
            
            # Put the processed embeddings back in their correct positions
            final_embeddings[batch_indices, timestep_indices, feature_indices] = llm_embeddings
        
        else:

            dummy_tokens = torch.zeros(1, max_tokens, dtype=torch.long, device=text_tokens.device)
            dummy_masks = torch.zeros(1, max_tokens, dtype=torch.long, device=text_tokens.device)
            _ = self.llm_module(token_ids=dummy_tokens, trace_grads=False, attention_mask=dummy_masks)


        return final_embeddings


    def _create_discriminator_input(
                self, 
                batch: ValueAssociatedTensorData, 
                gen_output: ValueAssociatedTensorData,
                record_masks: Dict[str, Dict[str, Tensor]]
            ) -> ValueAssociatedTensorData:
            """
            Create input for discriminator by replacing masked values with generated ones.
            
            Args:
                batch: Original value-associated data from the MixedTensorDataset batch input
                gen_output: Simulated value-associated data output from the MaskedTokenGenerator
                record_masks: Masks indicating which records and feature values were hidden from the generator. A value of 1
                    indicates a record was masked, 0 indicates that it was not.
                
            Returns:
                Updated batch with masked values replaced by generated ones
            """
            # Create a copy of feature value dicts to avoid modifying original when masking
            disc_input = deepcopy(batch)
            
            # Process each feature type
            for feat_type in ['numeric', 'categorical', 'text']:
                # The generator output only has keys for feature types that were used for prediction.
                # If the input batch's feature list for a type was empty, the output will not have a key for that type.
                # The exception is text, which is only processed if the user initialized the MaskedTokenGenerator with 
                # n_text_features > 0 (even if text features are present in the input batch).
                if feat_type not in gen_output or feat_type not in batch:
                    continue
                    
                if feat_type == 'numeric':
                    # Numeric values can be used as-is from generator
                    for i, (pred_vals, orig_vals) in enumerate(
                        zip(gen_output[feat_type]['values'], disc_input[feat_type]['values'])
                    ):
                        # Extract the value component mask for the ith feature
                        #     mask shape: (batch_size, max_ts_length, feature_dim)
                        value_mask = record_masks[feat_type]['values'][i].bool()
                        # Replace the values of masked feature value components with the generated values
                        disc_input[feat_type]['values'][i] = torch.where(value_mask, pred_vals, orig_vals)
                        
                elif feat_type == 'categorical':
                    # Convert generator logits to class indices for discriminator
                    for i, (pred_logits, orig_vals) in enumerate(
                        zip(gen_output[feat_type]['values'], disc_input[feat_type]['values'])
                    ):
                        # Extract the value component mask for the ith feature
                        value_mask = record_masks[feat_type]['values'][i].bool()
                        # Convert logits to class indices to resemble original data: softmax -> argmax
                        pred_probs = torch.softmax(pred_logits, dim=-1)
                        pred_classes = torch.argmax(pred_probs, dim=-1, keepdim=True).float()
                        # Replace the values of masked feature value components with the class indices
                        disc_input[feat_type]['values'][i] = torch.where(value_mask, pred_classes, orig_vals)
                        
                elif feat_type == 'text':
                    # Text values are embeddings have a consistent shape for all feature, so a single tensor is used.
                    #   Shape: (batch_size, max_timeseries_length, n_text_features, TEXT_EMBED_DIM).
                    pred_vals = torch.stack(gen_output[feat_type]['embedded_values'], dim=2)
                    orig_vals = disc_input[feat_type]['embedded_values']

                    # Extract the text embedding component mask and convert to boolean tensor
                    value_mask = torch.stack(record_masks[feat_type]['embedded_values'], dim=2).bool()

                    # Replace the values of masked feature value components with the generated values
                    disc_input[feat_type]['embedded_values'] = torch.where(value_mask, pred_vals, orig_vals)

                # Replace masked indicators with predicted ones if available
                if self.generator.predict_indicators:
                    # Ensure that the generator's indicators are strictly binary: 0 if value <= 0.5, else 1
                    pred_indicators = (gen_output[feat_type]['indicators'] > 0.5).float()
                    indicator_mask = record_masks[feat_type]['indicators'].bool()
                    disc_input[feat_type]['indicators'] = torch.where(
                        indicator_mask, pred_indicators,
                        disc_input[feat_type]['indicators']
                    )
            
            return disc_input
    
    def compute_conditional_intensity(self, encodings, prev_event_times, time_diff):
        """Wrapper to access hawkes submodule's method from parent model.
        
        For compatibility with model sharding using Accelerate.
        """
        return self.hawkes.compute_conditional_intensity(encodings, prev_event_times, time_diff)

    def compute_initial_intensity(self, batch_size):
        """Wrapper to access hawkes submodule's method from parent model.

        For compatibility with model sharding using Accelerate.
        """
        return self.hawkes.compute_initial_intensity(batch_size)

    def forward(
        self,
        batch: MixedTensorDataset,
        record_masks: Dict[str, Dict[str, List[List[Tensor]]]],
        device: str,
        trace_grads: bool = False,
        compute_intensities: bool = False,
        thp_loss_mc_samples: int = 100
    ) -> Dict[str, Union[Tensor, Tuple[Tensor, Tensor], Dict[str, Dict[str, Tensor]]]]:
        """
        Forward pass through the ELECTRA model.
        
        Args:
            batch: MixedTensorDataset containing value data, event data, static data, and targets
            record_masks: Masks indicating which records to simulate with generator
            device: The device where tensors will be sent.
            trace_grads: Whether to trace gradients through the LLM for XAI
            compute_intensities: Whether to compute conditional and initial intensities for the Hawkes process
            thp_loss_mc_samples: Number of Monte Carlo samples for THP loss estimation
            
        Returns:
            Dict containing event encodings, generator output, and discriminator predictions
        """
        outputs = {}
        
        # Process event data if available
        if 'event_data' in batch:
            event_enc, event_pred = self.hawkes(batch['event_data'])
            outputs['hawkes_encodings'] = event_enc
            outputs['hawkes_predictions'] = event_pred  # A tuple of (event_type_prediction, time_prediction)

            if compute_intensities:
                batch_size = event_enc.size(0)
                event_data = batch['event_data']
                event_times = event_data['times']
                event_non_padding_masks = event_data['masks']
                
                # Compute time differences
                time_diff_obs = calc_time_diff(event_times, event_non_padding_masks, device=device)
                # Sample inter-event time differences for Monte Carlo integration
                time_diff_samples = sample_non_event_time_diff(
                    time_diff_obs[:, 1:], n=thp_loss_mc_samples, device=device
                )
                
                # Compute intensity values
                obs_initial_intensities = self.hawkes.compute_initial_intensity(batch_size)
                obs_conditional_intensities = self.hawkes.compute_conditional_intensity(
                    encodings=event_enc[:, :-1, :],
                    prev_event_times=event_times[:, :-1],
                    time_diff=time_diff_obs[:, 1:]
                )
                sampled_intensities = self.hawkes.compute_conditional_intensity(
                    encodings=event_enc[:, :-1, :],
                    prev_event_times=event_times[:, :-1],
                    time_diff=time_diff_samples
                )
                
                # Store intensity values in outputs
                outputs['thp_intensities'] = {
                    'obs_initial': obs_initial_intensities,
                    'obs_conditional': obs_conditional_intensities,
                    'sampled': sampled_intensities,
                    'time_diff_obs': time_diff_obs,
                    'time_diff_samples': time_diff_samples
                }
            
        value_data = batch['val_data']  # Extract the ValueAssociatedTensorData from the batch

        if self.use_text and 'text' in value_data:

            # Process text data through LLM
            text_embeddings = self._gen_text_embeddings(value_data, trace_grads)

            # Store embeddings in processed batch
            value_data['text']['embedded_values'] = text_embeddings

        # Generate predictions for masked values
        gen_output = self.generator(value_data, record_masks)
        outputs['generator'] = gen_output
        
        # Create input for discriminator by replacing masked values with generated ones
        disc_input = self._create_discriminator_input(value_data, gen_output, record_masks)
        
        # Get discriminator predictions
        disc_output = self.discriminator(disc_input, batch.get('static_data', None))
        outputs['discriminator'] = disc_output

        return outputs
    

class MixedClassifier(torch.nn.Module):
    """A classifier that combines event and value-associated time series data for prediction.
    
    This model processes both event sequences and value-associated data through separate encoders,
    aggregates their outputs, optionally incorporates static data, and makes final predictions.
    """
    
    def __init__(
        self, 
        event_encoder: EventDataEncoder,
        val_encoder: ValueDataEncoder,
        d_event_enc: int, 
        d_val_enc: int,
        d_statics: int,
        num_classes: int,
        aggr: str = 'max',
        use_text: bool = False,
        llm_module: Optional[GradientTraceableLLM] = None
    ):
        """Initialize MixedClassifier.
        
        Args:
            event_encoder: Encoder for event-associated data
            val_encoder: Encoder for value-associated time series data
            d_event_enc: Dimensionality of event encoder output
            d_val_enc: Dimensionality of time series encoder output
            d_statics: Dimensionality of static data (0 if no static data)
            num_classes: Number of output classes
            aggr: Aggregation method ('max' or 'mean') for sequence-level encoding
            use_text: If True, the model will be initialized with a GradientTraceableLLM instance, and the model will   
                expect text features in the input.
            llm_module: Optional pre-initialized GradientTraceableLLM instance to use for text processing. If None and 
                use_text is True, a new instance will be created. This can be used to share the same LLM module across different models (for example, when using a pretrained, frozen LLM), saving a significant amount of memory.
        """

        super().__init__()
        self.event_encoder = event_encoder
        self.val_encoder = val_encoder
        self.linear = torch.nn.Linear(d_event_enc + d_val_enc + d_statics, 32)
        self.linear1 = torch.nn.Linear(32, num_classes)
        self.aggr = aggr
        self.use_text = use_text

        # Use shared LLM if use_text and llm_module provided
        if self.use_text:
            if llm_module is not None:
                self.llm_module = llm_module
            else:
                self.llm_module = GradientTraceableLLM()
        else:
            self.llm_module = None

    def _process_text_embeddings(
        self, 
        value_data: Dict, 
        trace_grads: bool = False
    ) -> torch.Tensor:
        """Same method as in ELECTRA - process text through LLM efficiently."""
        # Copy the exact _gen_text_embeddings method from ELECTRA
        text_tokens = torch.stack(value_data['text']['values'], dim=2)
        text_token_masks = torch.stack(value_data['text']['masks'], dim=2)
        
        batch_size, max_ts_len, n_text_feats, max_tokens = text_tokens.shape
        
        # Create a mask to identify timesteps with actual text data
        has_text_per_timestep = (text_tokens.max(dim=-1)[0] > 0)
        has_any_text = has_text_per_timestep.any()
        
        # Initialize embeddings tensor with zeros
        final_embeddings = torch.zeros(
            batch_size, max_ts_len, n_text_feats, TEXT_EMBED_DIM,
            device=text_tokens.device, 
            dtype=torch.float32
        )
        
        # If text is present, perform a forward pass only on those positions. If text is not present anywhere in the
        # batch, perform a forward pass with a dummy tensor to avoid NCCL timeouts due to desynchronization when using
        # multiple GPUs in distributed computation.
        if has_any_text:
            # Initialize embeddings tensor with zeros
            final_embeddings = torch.zeros(
                batch_size, max_ts_len, n_text_feats, TEXT_EMBED_DIM,
                device=text_tokens.device, 
                dtype=torch.float32
            )
            
            # Find positions where we have actual text
            text_positions = torch.where(has_text_per_timestep)
            
            # Extract only the tokens that have actual text
            batch_indices, timestep_indices, feature_indices = text_positions
            
            # Get the actual text tokens and masks for processing
            tokens_to_process = text_tokens[batch_indices, timestep_indices, feature_indices]
            masks_to_process = text_token_masks[batch_indices, timestep_indices, feature_indices]

            # Process through LLM
            llm_embeddings = self.llm_module(
                token_ids=tokens_to_process, 
                trace_grads=trace_grads, 
                attention_mask=masks_to_process
            )

            # Put the processed embeddings back in their correct positions
            final_embeddings[batch_indices, timestep_indices, feature_indices] = llm_embeddings
        else:
            dummy_tokens = torch.zeros(1, max_tokens, dtype=torch.long, device=text_tokens.device)
            dummy_masks = torch.zeros(1, max_tokens, dtype=torch.long, device=text_tokens.device)
            _ = self.llm_module(token_ids=dummy_tokens, trace_grads=False, attention_mask=dummy_masks)

        return final_embeddings

    def forward(self, batch: MixedTensorDataset, trace_grads: bool = False) -> Tensor:
        """Forward pass through the mixed classifier.
        
        Args:
            batch (MixedTensorDataset): MixedTensorDataset containing event data, value data, and optionally static data
            trace_grads (bool): Whether to trace gradients through the LLM

        Returns:
            Tensor: Classification logits of shape (batch_size, num_classes)
        """

        embeddings = []
        
        # Process event data if available
        if 'event_data' in batch:

            event_data = batch['event_data']

            event_indicators = event_data['indicators']
            event_times = event_data['times']
            event_masks = event_data['masks']
            
            # Pass the event data through the encoder
            #     Shape: (batch_size, max_ts_length, d_event_enc)
            event_enc = self.event_encoder(event_indicators, event_times, event_masks)
            
            # Aggregate event encodings across the time dimension (dim=1)
            #     event_enc final shape: (batch_size, d_event_enc)
            event_enc = event_enc * event_masks[..., None].float()  # Zero out padding embeddings   
            n_obs_records = event_masks.sum(dim=-1, keepdim=True).clamp(min=1)  # Clamp to avoid errors

            if self.aggr == 'max':
                event_enc, _ = torch.max(event_enc, dim=1)
            elif self.aggr == 'mean':
                event_enc = torch.sum(event_enc, dim=1) / n_obs_records
            elif self.aggr == 'none':
                # Select the final observed record's encoding for each batch item
                # If a batch item has no observed records, use the zeroed embedding at the first timestep
                final_record_idx = n_obs_records.squeeze(-1) - 1  # (batch_size, )
                batch_size = event_enc.size(0)
                event_enc = event_enc[torch.arange(batch_size), final_record_idx.long()]
            
            embeddings.append(event_enc)
        
        # Process value-associated data if available
        if 'val_data' in batch:

            val_data = batch['val_data']

            val_times = val_data['times']
            val_masks = val_data['masks']
            
            if self.use_text and 'text' in val_data:
                if 'embedded_values' not in val_data['text']:
                    # Generate text embeddings
                    text_embeddings = self._process_text_embeddings(val_data, trace_grads)
                    val_data['text']['embedded_values'] = text_embeddings

            # Combine all feature types along a single axis for the encoder
            inds_to_concat = []
            vals_to_concat = []
            
            # Process numeric features
            if 'numeric' in val_data and val_data['numeric']['values']:
                # Extract numeric feature indicators
                numeric_inds = val_data['numeric']['indicators']
                inds_to_concat.append(numeric_inds)
                # Extract and concatenate numeric feature values
                numeric_vals = torch.cat(val_data['numeric']['values'], dim=2)
                vals_to_concat.append(numeric_vals)
            
            # Process categorical features
            if 'categorical' in val_data and val_data['categorical']['values']:
                # Extract categorical feature indicators
                categorical_inds = val_data['categorical']['indicators']
                inds_to_concat.append(categorical_inds)
                # Extract and concatenate categorical feature values
                categorical_vals = torch.cat(val_data['categorical']['values'], dim=2)
                vals_to_concat.append(categorical_vals)

            # Process text features
            if 'text' in val_data and 'embedded_values' in val_data['text']:
                # Extract text feature indicators and embeddings
                text_inds = val_data['text']['indicators']
                inds_to_concat.append(text_inds)
                text_embeddings = val_data['text']['embedded_values'].flatten(start_dim=2)
                vals_to_concat.append(text_embeddings)
        
            if inds_to_concat and vals_to_concat:
                # Concatenate the tensors for numeric and categorical features along the feature dimension
                #   combined_val_indicators shape: (batch_size, max_timeseries_length, n_num_feats + n_cat_feats)
                #   combined_val_values shape: (batch_size, max_timeseries_length, total_feat_dim)
                combined_val_indicators = torch.cat(inds_to_concat, dim=2)
                combined_val_values = torch.cat(vals_to_concat, dim=2)
                
                # Pass through time series encoder
                val_enc = self.val_encoder(
                    combined_val_indicators, 
                    combined_val_values, 
                    val_times, 
                    val_masks
                )
                
                # Aggregate time series encodings across the time dimension (dim=1)
                #     val_enc final shape: (batch_size, d_val_enc)
                val_enc = val_enc * val_masks[..., None].float() # Zero out padding embeddings
                n_obs_records = val_masks.sum(dim=-1, keepdim=True).clamp(min=1)  # Clamp to avoid errors

                if self.aggr == 'max':
                    val_enc, _ = torch.max(val_enc, dim=1)
                elif self.aggr == 'mean':
                    # Exclude padding timesteps from mean calculation
                    val_enc = torch.sum(val_enc, dim=1) / n_obs_records
                elif self.aggr == 'none':
                    # Select the final observed record's encoding for each batch item
                    # If a batch item has no observed records, use the zeroed embedding at the first timestep
                    final_record_idx = n_obs_records.squeeze(-1) - 1  # (batch_size, )
                    batch_size = val_enc.size(0)
                    val_enc = val_enc[torch.arange(batch_size), final_record_idx.long()]

                embeddings.append(val_enc)
        
        # Add static data if available
        if 'static_data' in batch and batch['static_data'] is not None:
            embeddings.append(batch['static_data'])
        
        # Combine all embeddings
        if len(embeddings) > 1:
            enc = torch.cat(embeddings, dim=1)
        else:
            enc = embeddings[0]
        
        # Final classification layers
        enc = self.linear(enc)
        return self.linear1(torch.nn.functional.gelu(enc))
    