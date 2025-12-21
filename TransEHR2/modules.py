import torch

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple

from TransEHR2.constants import HF_API_TOKEN, LLM_NAME, MAX_TOKEN_LENGTH, PAD, TEXT_EMBED_DIM, TOKENIZER_PAD_TOKEN
from TransEHR2.data.custom_types import EventAssociatedTensorData, ValueAssociatedTensorData
from TransEHR2.layers import EncoderLayer, TemporalPositionEncoding, TransformerBatchNormEncoderLayer
from TransEHR2.utils import combine_value_and_text_data


class GradientTraceableLLM(torch.nn.Module):
    """A wrapper for a language model that allows gradients to be traced through it."""

    def __init__(self, model_name: str = LLM_NAME, max_length: int = MAX_TOKEN_LENGTH):

        super().__init__()
        # Initialize the model on CPU to avoid GPU memory issues during FSDP wrapping
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=HF_API_TOKEN, 
            device_map='cpu'
        )
        # Using add_special_tokens so that pad_token_id is set automatically
        self.tokenizer.add_special_tokens({'pad_token': TOKENIZER_PAD_TOKEN})
        self.model = AutoModel.from_pretrained(
            model_name,
            token=HF_API_TOKEN,
            device_map='cpu'
            # Note: Using default dtype (float32) to ensure consistency with FSDP
        )
        # Resize to account for padding token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = max_length

        # Freeze the LLM parameters to prevent them from being updated during training
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, 
        token_ids: Tensor, 
        trace_grads: bool = False, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through the LLM.
        
        Args:
            
            token_ids (Tensor): A tensor of N token ID sequences with shape [N, max_token_length].
            trace_grads (bool, optional): Whether to trace gradients through the LLM. Feature attribution requires 
                that `trace_grads` be set to `True`. Defaults to `False`.
            attention_mask (Tensor, optional): A tensor of token length padding masks with shape [N, max_token_length]. 
                Defaults to None.
        
        Returns:
            Tensor: A tensor of shape [N, llm_embedding_dim] of LLM embeddings of input token sequences.
        """

        # Ensure token_ids are integers
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        
        if attention_mask is not None and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        # Store token sequence inputs for gradient attribution
        self.last_token_ids = token_ids

        with torch.set_grad_enabled(trace_grads):  # Only trace gradients if trace_grads is True
            outputs = self.model(token_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # (N, max_token_length, embed_dim)
            # Mean-pool the non-padding token embeddings of each sequence
            #   embeddings shape after mean-pooling: (N, embed_dim)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                embeddings = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                embeddings = embeddings.mean(dim=1)
            
            # Add a hook to save gradients during the backward pass, but only if we're tracing gradients
            if trace_grads and embeddings.requires_grad:
                embeddings.register_hook(self._save_gradients)
        
        return embeddings
    
    def _save_gradients(self, grad: Tensor) -> None:
        """Save gradients during the backward pass."""
        self.embedding_gradients = grad

    def attribute():
        # TODO: Implement text feature attributions to predictions at a token level.
        #   Should return a tensor of scores with the same shape as `token_ids` from the forward pass.
        raise NotImplementedError()  
    
    def get_attribution_text(self, token_ids: Tensor, scores: Tensor) -> List[List[List[List[Tuple[str, float]]]]]:
        """Convert attribution scores to human-readable text with highlights.
                
        Args:
            token_ids (Tensor): Tensor of token IDs with shape (batch_size, max_timeseries_length, n_text_features, max_token_length)
            scores (Tensor): Attribution scores with matching shape
                
        Returns:
            List[List[List[List[Tuple[str, float]]]]]: Nested lists representing decoded tokens with scores:
                - First level: batch dimension
                - Second level: timestep (record) dimension
                - Third level: text features dimension
                - Fourth level: list of (token, score) pairs for each token in the text feature
        """

        if token_ids.shape != scores.shape:
            raise ValueError(
                f'Expected token_ids and scores to have the same shape, got {token_ids.shape} and {scores.shape}.'
            )

        if token_ids.dim() != 4:
            if token_ids.dim() == 3:
                # Assume shape (max_timeseries_length, n_features, max_token_length), add batch dim of size 1
                token_ids = token_ids.unsqueeze(0)
                scores = scores.unsqueeze(0)  # Ensure scores also have batch dimension
            else:
                raise ValueError(f'Expected token_ids to be 3D or 4D, got {token_ids.dim()}D tensor.')

        results = []
        for batch_idx in range(token_ids.shape[0]):
            timeseries_results = []
            for time_idx in range(token_ids.shape[1]):
                feature_results = []
                for feature_idx in range(token_ids.shape[2]):
                    sample_tokens = token_ids[batch_idx, time_idx, feature_idx]
                    sample_scores = scores[batch_idx, time_idx, feature_idx]
                    # Convert token IDs to text with score annotations
                    decoded_tokens = self.tokenizer.convert_ids_to_tokens(sample_tokens)
                    token_score_pairs = [(tk, sc.item()) for tk, sc in zip(decoded_tokens, sample_scores) 
                                            if tk != self.tokenizer.pad_token]
                    feature_results.append(token_score_pairs)
                timeseries_results.append(feature_results)
            results.append(timeseries_results)

        return results


class EventDataEncoder(torch.nn.Module): 
    # NOTE Originally from https://github.com/SimiaoZuo/Transformer-Hawkes-Process/blob/master/transformer/Models.py,
    # specifically the Encoder class.
    """ A encoder model with self attention mechanism. 
    
    This model is used to encode event sequences, which are then used to predict the next event type and time.

    Attributes:
        d_model (int): The dimensionality of the model.
        position_vec (Tensor): Positional vector used for temporal encoding.
        event_emb (nn.Embedding): Layer that embeds event types into a `d_model`-dimensional space.
        layer_stack (nn.ModuleList): A list of `n_layers` `EncoderLayer` instances.
    """

    def __init__(
            self,
            num_types: int,
            d_model: int, 
            d_inner: int,
            n_layers: int, 
            n_head: int, 
            d_k: int, 
            d_v: int,
            dropout: float
    ):
        """Initialize an instance.

        Args:
            num_types (int): The number of unique event types in the dataset.
            d_model (int): The dimensionality of the model.
            d_inner (int): The dimensionality of the inner layers.
            n_layers (int): The number of encoder layers.
            n_head (int): The number of attention heads.
            d_k (int): The dimensionality of the key vectors.
            d_v (int): The dimensionality of the value vectors.
            dropout (float): The dropout rate applied throughout the encoder.
        """

        super().__init__()
        self.num_types = num_types
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        
        # NOTE Xu et al. used a torch.nn.Embedding layer to project event types to the model dimension. That worked
        # because the original implementation of the forward pass expected one event ID per timestep (i.e. [batch_size, 
        # seq_length]), and there could be duplicate timestamps when more than one type of event occurred at the same 
        # time. However, this is not how they described the input structure in the paper, and our implementation 
        # represents events as a binary indicator vector with shape (batch_size, sequence_length, num_types) to reflect 
        # the methods described in the paper. Instead of using an Embedding layer, we use a torch.nn.Linear layer as 
        # described in Xu et al.
        self.indicator_input_projection_layer = torch.nn.Linear(self.num_types, self.d_model, bias=False)
        # NOTE Xu et al. applied dropout at 0.1 to the timeseries positional encoding for the value-associated data
        # encoder, but not to the position encoding of the event-associated data encoder. To keep things consistent,
        # dropout will also be applied to the position encoding of the event-associated data encoder in this 
        # implementation.
        self.position_encoding_layer = TemporalPositionEncoding(d_model=self.d_model, dropout=0.1)
        enc_args, enc_kwargs = (
            [self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v],
            {'dropout': self.dropout, 'normalize_before': False}
        )
        self.layer_stack = torch.nn.ModuleList([EncoderLayer(*enc_args, **enc_kwargs) for _ in range(self.n_layers)])

    @staticmethod
    def _get_subsequent_mask(input_sequence: Tensor) -> Tensor:
        """Masks future tokens for each token in an input sequence of tokens.

        Args:
            input_sequence (Tensor): A `Tensor` of shape [batch_size, max_timeseries_length, n_event_types].
        
        Returns:
            Tensor: A `Tensor` of shape [batch size, sequence length, sequence length] indicating which timesteps are 
                masked (1) and which ones are not (0). The mask matrix is an upper triangular matrix with zeros on the diagnonal.
        """

        batch_size, max_ts_len, _ = input_sequence.size()
        # Create an upper triangular boolean mask for subsequent positions
        mask = torch.ones((max_ts_len, max_ts_len), device=input_sequence.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # Convert to upper triangular matrix with zeros on the diagonal
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch size, seq len, seq len]

        return mask

    @staticmethod
    def _get_attn_key_pad_mask(timeseries_mask: Tensor) -> Tensor:
        """For masking out the padding part of key sequence
        
        Given the key sequence and the query sequence, this function returns a mask tensor that indicates which tokens in the key sequence are padding tokens.


        Args:
            seq_k (Tensor): Key sequence `Tensor`
            seq_q (Tensor): Query sequence `Tensor`
        
        Returns:
            Tensor: A boolean `Tensor` of shape [batch size, q sequence length, k sequence length] indicating which 
                tokens in the key sequence are padding tokens.
        """

        # NOTE: originally the input was a (batch_size, max_timeseries_length) tensor of event integer IDs, but now it
        # is a (batch_size, max_timeseries_length, n_event_types) tensor of indicator vectors. An event ID of zero 
        # formerly meant padding, so the original implementation would probably have masked not only padding timesteps, 
        # but also timesteps where no events occurred. In this new implementation the latter is not possible, since we
        # only provide a mask tensor for timeseries length padding. This is probably a good thing since the absence of
        # events may be informative.
        max_timeseries_length = timeseries_mask.size(1)  # Maximum length of the timeseries
        padding_mask = timeseries_mask.eq(PAD)  # True where the timeseries is length-padded
        padding_mask = padding_mask.unsqueeze(1).expand(-1, max_timeseries_length, -1)  # b x lq x lk
        padding_mask = padding_mask.type(torch.bool)

        return padding_mask
    
    def forward(
        self,
        indicators: Tensor,
        timestamps: Tensor,
        non_padding_mask: Tensor
    ) -> Tensor:
        """ Encode event sequences via masked self-attention. 
        
        Args:
            indicators (Tensor): A (batch_size, max_timeseries_length, n_event_types) tensor containing the event 
                indicator vectors.
            timestamps (Tensor): A (batch size, max_timeseries_length) tensor containing the event timestamps.
            non_padding_mask (Tensor): A (batch size, max_timeseries_length) tensor indicating which tokens are 
                part of the input sequence as opposed to padding tokens.
        
        Returns:
            Tensor: A (batch size, max_timeseries_length, d_model) tensor containing the encoded event sequences.

        Note:
            The initial embedding is obtained by applying the indicator input projection layer to the indicator vectors.
        """

        # Create a mask to prevent attention to future timesteps at each timestep
        # The mask has shape (batch_size, max_timeseries_length, max_timeseries_length)
        subsequent_mask = self._get_subsequent_mask(indicators)
        # Mask the key vectors to prevent attention to padding tokens
        attn_key_pad_mask = self._get_attn_key_pad_mask(non_padding_mask)
        # Combine the key padding and future event masks
        self_attn_mask = (attn_key_pad_mask + subsequent_mask).gt(0)

        # Apply the indicator input projection layer to the indicators to get the initial embedding
        enc_output = self.indicator_input_projection_layer(indicators.float())
        for enc_layer in self.layer_stack:
            position_encoding = self.position_encoding_layer(enc_output, timestamps, non_padding_mask)
            enc_output += position_encoding
            enc_output, _ = enc_layer(enc_output, non_padding_mask=non_padding_mask, self_attention_mask=self_attn_mask)

        return enc_output


class ValueDataEncoder(torch.nn.Module):
    r"""Transformer encoder network for time series data.

    A PyTorch implementation of a Transformer encoder specifically designed for time series data. This module 
    implements a standard Transformer encoder architecture with support for both fixed and learnable positional 
    encodings, different normalization schemes, and various activation functions.

    Attributes:
        d_model (int): Dimension of the model's hidden layers and embedding vectors.
        n_heads (int): Number of attention heads in multi-head attention layers.
        project_inp (torch.nn.Linear): Linear projection layer to transform input features to model dimension.
        pos_enc (PositionalEncoding): Positional encoding layer (either fixed or learnable).
        transformer_encoder (torch.nn.TransformerEncoder): Main transformer encoder stack.
        act (torch.nn.ReLU|torch.nn.GELU): Activation function used throughout the network.
        dropout1 (torch.nn.Dropout): Dropout layer applied to outputs.
        feat_dim (int): Dimension of input features.
    """

    def __init__(
        self,
        n_features: int,
        feat_dim: int,
        d_model: int, 
        n_heads: int, 
        n_encoder_blocks: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm: str = 'BatchNorm',
    ):
        r"""Initialize an instance.

        Args:
            n_features (int): The number of unique features in the input data. This is the size of the feature 
                indicator vector.
            feat_dim (int): The combined dimensionality of all the input features' values. This is the size of the 
                feature value vector.
            d_model (int): The dimensionality of the model's embedding.
            n_heads (int): The number of attention heads.
            n_encoder_blocks (int): The number of encoder blocks to use.
            dim_feedforward (int): The size of the hidden layers in the feedfoward network.
            dropout (float, optional): Rate of dropout applied throughout the network. Defaults to 0.1.
            pos_encoding (str, optional): The strategy to use for generating a positional encoding of a timestamp.
                If 'learnable', use `LearnablePositionalEncoding`. If 'fixed', use `FixedPositionalEncoding`. Defaults to 'fixed'.
            activation (str, optional): The activation function applied throughout the network. Defaults to 'gelu'.
                norm (str, optional): The type of normalization to use in the `TransformerEncoder`. If 'LayerNorm', `self.transformer_encoder` will be initialized with an instance of `TransformerEncoder` that uses `TransformerEncoderLayer'. Otherwise, `self.transformer_encoder` will be initialized with an instance of `TransformerEncoder` that uses `TransformerBatchNormEncoderLayer`.
        """

        super().__init__()
        self.n_features = n_features
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_blocks = n_encoder_blocks
        self.dim_feedforward = dim_feedforward
        self.norm = norm
        # NOTE Xu et al. only used the value_input_projection_layer in their code, but in the paper they described
        # using two separate projection layers for the indicator and value-associated data. We follow the paper.
        # The value input projection uses bias, but the indicator input projection does not.
        self.indicator_input_projection_layer = torch.nn.Linear(n_features, d_model, bias=False)
        self.value_input_projection_layer = torch.nn.Linear(feat_dim, d_model, bias=True)
        # NOTE Xu et al. applied dropout at 0.1 to the timeseries positional encoding for the value-associated data
        # encoder, but not to the position encoding of the event-associated data encoder. To keep things consistent,
        # dropout will also be applied to the position encoding of the event-associated data encoder in this 
        # implementation.
        self.position_encoding_layer = TemporalPositionEncoding(d_model=d_model, dropout=0.1)
        self.encoder_layer = self._get_encoder_layer(norm, activation, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_blocks)
        self.activation = self._get_activation_layer(activation)
        self.dropout = torch.nn.Dropout(dropout)


    def _get_encoder_layer(self, norm: str, activation: str, dropout: float) -> torch.nn.Module:
        """Get the encoder layer to use in the transformer."""

        enc_args, enc_kwargs = (
            [self.d_model, self.n_heads, self.dim_feedforward, dropout],
            {'activation': activation, 'batch_first': True}
        )
        if norm == 'LayerNorm':
            return torch.nn.TransformerEncoderLayer(*enc_args, **enc_kwargs)
        elif norm == 'BatchNorm':
            return TransformerBatchNormEncoderLayer(*enc_args, **enc_kwargs)
        else:
            raise ValueError(f'norm: Expected "LayerNorm" or "BatchNorm", got {norm}.')


    @staticmethod
    def _get_activation_layer(activation):
        if activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'gelu':
            return torch.nn.GELU()
        else:
            raise ValueError(f'activation: expected "relu" or "gelu", got {activation}')


    def forward(self, indicators: Tensor, values: Tensor, timestamps: Tensor, timestep_masks: Tensor) -> Tensor:
        """Pass the input through the transformer network.

        The forward pass performs a linear projection of the input features to the model dimension, adds a 
        time-dependent positional encoding to give the final embedding of the input sequence, and passes it through the 
        transformer encoder.

        Note:
            The output of this method is not passed through any output head or final projection layer; it returns the 
            raw transformer embeddings with shape [batch_size, seq_length, self.d_model]. Users expecting a final 
            projection (e.g., for classification or regression) should apply an output head separately.

        Args:
            indicators (Tensor): A (batch_size, max_timeseries_length, n_features) tensor of bits indicating which 
                features are recorded at each time step.
            values (Tensor): A (batch_size, max_timeseries_length, total_feat_dim) tensor of feature values, where 
                total_feat_dim is the sum total of the number of components of all features.
            timestamps (Tensor): A (batch_size, max_timeseries_length) tensor of timestamps for each timestep (record) 
                in the input sequence.
            timestep_masks (Tensor): A (batch_size, max_timeseries_length) tensor of bits indicating which records
                are part of the data (1) and which ones are padding (0). Note that this is the opposite of the 
                convention used by `torch.nn.modules.MultiheadAttention` in `TransformerEncoderLayer`.
        Returns:
            Tensor: Returns the result of a forward pass through the transformer network. The output Tensor
                has shape [batch_size, seq_length, self.d_model].
        """

        # Apply a linear transformation to the value and indicator inputs and add them together
        # NOTE Xu et al. only used the value_input_projection_layer in their code, but in the paper they described
        # using two separate projection layers for the indicator and value-associated data. We follow the paper.
        indicator_projection = self.indicator_input_projection_layer(indicators.float())
        value_projection = self.value_input_projection_layer(values)
        embedding = indicator_projection + value_projection
        # Add time-dependent positional embeddings
        embedding = self.position_encoding_layer(embedding, timestamps, timestep_masks)

        # Permute the embedding tensor to match the expected input shape for the transformer encoder
        # The transformer expects the input in the shape (seq_length, batch_size, d_model)
        embedding = embedding.permute(1, 0, 2)
        
        # Perform the forward pass through the transformer encoder
        # NOTE padding mask logic is reversed to comply with MultiHeadAttention, TransformerEncoderLayer
        # Expects 0/False for tokens that are not masked, 1/True for padding tokens that are masked
        inverted_timestep_masks = ~timestep_masks.bool().T  # Shape: (seq_length, batch_size)
        embedding = self.transformer_encoder(embedding, src_key_padding_mask=inverted_timestep_masks)  
        embedding = self.activation(embedding)
        embedding = self.dropout(embedding)
        
        # Return to original dim order, (batch_size, seq_length, d_model)
        embedding = embedding.permute(1, 0, 2) 

        return embedding


class MaskedTokenGenerator(torch.nn.Module):
    """
    A masked language model for time series data prediction that can predict both
    feature values and, optionally, feature presence indicators.
    """

    def __init__(
        self, 
        encoder: torch.nn.Module,
        d_model: int, 
        numeric_dims: List[int], 
        categorical_classes: List[int], 
        n_text_features: int = 0,
        predict_indicators: bool = False,
        dim_feedforward: Optional[int] = 128
    ):
        """Initialize an instance

        Args:
            encoder (torch.nn.Module): The transformer encoder module
            d_model (int): The dimensionality of the model's embedding vectors
            numeric_dims (List[int]): List of dimensions for each numeric feature
            categorical_classes (List[int]): List of class counts for each categorical feature
            n_text_features (int): Number of text features. Defaults to 0. If 0, no text features will be processed or 
                predicted.
            predict_indicators (bool): Whether to predict feature presence indicators
            dim_feedforward (int, optional): Used when predict_indicators is True. The dimensionality of the hidden 
                layer in the MLP that predicts indicators. Defaults to 128.
        """

        super().__init__()
        self.encoder = encoder

        # In the original implementation that only considered scalar real-valued features, the output head was a single 
        # linear layer that produced a tensor with n_features as the last dimension. In this implementation, there is a 
        # separate output head for each feature type.

        self.numeric_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, feat_dim) for feat_dim in numeric_dims
        ])
        self.categorical_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, n_classes) for n_classes in categorical_classes
        ])
        self.text_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, TEXT_EMBED_DIM) for _ in range(n_text_features)
        ])

        self.predict_numeric_feats = len(self.numeric_heads) > 0
        self.predict_categorical_feats = len(self.categorical_heads) > 0
        self.predict_text_feats = len(self.text_heads) > 0

        # Indicator prediction heads (optional)
        self.predict_indicators = predict_indicators  # If True, the model will also predict feature presence indicators
        if self.predict_indicators:
            self.indicator_mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, dim_feedforward),
                torch.nn.GELU()
            )
            self.numeric_indicator_head = torch.nn.Linear(dim_feedforward, len(numeric_dims))
            self.categorical_indicator_head = torch.nn.Linear(dim_feedforward, len(categorical_classes))
            self.text_indicator_head = torch.nn.Linear(dim_feedforward, n_text_features)

    def forward(
        self, 
        batch: ValueAssociatedTensorData,
        record_masks: Dict[str, Dict[str, List[List[Tensor]]]]
    ) -> ValueAssociatedTensorData:
        """
            Perform a forward pass through the model to predict masked features.
            
            Args:
                batch: A batch of Tensors prepared from the MixedDataset structure, which may include numeric 
                    features with indicators and values, categorical features with indicators and values, and text 
                    features with indicators and token IDs.
                record_masks: A dictionary of value-associated data masks for each feature type (returned by 
                    utils.generate_record_masks()). The masked values are simulated by the generator and used for 
                    training. Masked components should be represented by values of one, while unmasked components 
                    should be represented by values of zero.
                
            Returns:
                A dictionary (ValueAssociatedTensorData) with the following structure:
                {
                    'numeric': {
                        'indicators': 
                            tensor(batch_size, max_ts_len, n_numeric_feats)  # Feature indicator bits
                        'values': [  # List of feature tensors
                            tensor(batch_size, max_ts_len, feature_dim),  # Feature 1 values
                            tensor(batch_size, max_ts_len, feature_dim),  # Feature 2 values
                            ...  # More features
                        ]
                    },
                    'categorical': { ... },
                    'text': { ... }  # feature_dim for text is TEXT_EMBED_DIM
                }
                'times': tensor(batch_size, max_ts_len)  # Timestamps for each record, not predicted but passed through
                'masks': tensor(batch_size, max_ts_len)  # Masks for each record, not predicted but passed through

                Note that the 'indicators' lists are only returned if the MaskedTokenGenerator is initialized to produce them alongside the values. Each data type key is only present if there is at least one feature of that type in the dataset.
        """

        inds_to_concat = []
        vals_to_concat = []

        # Extract a list of tensors for each feature type; these will be concatenated along the feature dimension
        if self.predict_numeric_feats:
            # Extract and mask numeric feature indicators
            numeric_inds = batch['numeric']['indicators']
            masked_numeric_inds = numeric_inds * record_masks['numeric']['indicators']
            inds_to_concat.append(masked_numeric_inds)
            # Extract and mask numeric feature values
            numeric_vals = torch.cat(batch['numeric']['values'], dim=2)
            masked_numeric_vals = numeric_vals * torch.cat(record_masks['numeric']['values'], dim=2)
            vals_to_concat.append(masked_numeric_vals)

        if self.predict_categorical_feats:
            # Extract and mask categorical feature indicators
            categorical_inds = batch['categorical']['indicators']
            masked_categorical_inds = categorical_inds * record_masks['categorical']['indicators']
            inds_to_concat.append(masked_categorical_inds)
            # Extract and mask categorical feature values
            categorical_vals = torch.cat(batch['categorical']['values'], dim=2)
            masked_categorical_vals = categorical_vals * torch.cat(record_masks['categorical']['values'], dim=2)
            vals_to_concat.append(masked_categorical_vals)

        # Concatenate the tensors for numeric and categorical features along the feature dimension
        #   masked_indicators shape: (batch_size, max_timeseries_length, n_num_feats + n_cat_feats)
        #   masked_values shape: (batch_size, max_timeseries_length, total_feat_dim)
        masked_indicators = torch.cat(inds_to_concat, dim=2)
        masked_values = torch.cat(vals_to_concat, dim=2)

        timestamps = batch['times']  # (batch_size, max_timeseries_length)
        timestep_masks = batch['masks']  # (batch_size, max_timeseries_length)

        if self.predict_text_feats:
            # Extract text feature indicators, token sequences, and token sequence attention masks from the batch
            #   text_indicators shape: [batch_size, max_timeseries_length, n_text_features]
            #   text_embeddings shape: [batch_size, max_timeseries_length, n_text_features, TEXT_EMBED_DIM]
            text_indicators = batch['text']['indicators']
            text_embeddings = batch['text']['embedded_values']
            # Zero out masked text feature embedding components
            masked_text_indicators = text_indicators * record_masks['text']['indicators']
            masked_text_embeddings = text_embeddings * torch.stack(record_masks['text']['embedded_values'], dim=2)
            # Combine the numeric, categorical, and text data into a single tensor for input to the encoder
            #   masked_indicators shape: (batch_size, max_timeseries_length, n_num_feats + n_cat_feats + n_text_feats)
            #   masked_values shape: (batch_size, max_timeseries_length, total_feat_dim)
            masked_indicators, masked_values = combine_value_and_text_data(
                value_assoc_indicators=masked_indicators,
                value_assoc_values=masked_values,
                text_assoc_indicators=masked_text_indicators, 
                text_embeddings=masked_text_embeddings
            )

        # Get embeddings for each batch item, timestep from the encoder transformer
        if self.predict_indicators:
            indicator_embed = self.indicator_mlp(masked_indicators)  # (batch_size, max_timeseries_length, d_model)
        # val_embed shape: (batch_size, max_timeseries_length, d_model)
        val_embed = self.encoder(
            masked_indicators,  # [batch_size, max_timeseries_length, n_num_feats + n_cat_feats + n_text_feats]
            masked_values,  # [batch_size, max_timeseries_length, total_feat_dim]
            timestamps,  # [batch_size, max_timeseries_length]
            timestep_masks  # [batch_size, max_timeseries_length]
        )

        # Initialize the output structure for each feature type
        output = {}

        if self.predict_numeric_feats:
            output['numeric'] = {'indicators': None, 'values': []}
            if self.predict_indicators:
                # Indicator output shape: [batch_size, max_timeseries_length, n_numeric_feats]
                output['numeric']['indicators'] = self.numeric_indicator_head(indicator_embed)
            # Each head produces predictions for one numeric feature
            # Value output shape: [(batch_size, max_timeseries_length, feature_dim). n_numeric_feats]
            output['numeric']['values'] = [head(val_embed) for head in self.numeric_heads]
        
        if self.predict_categorical_feats:
            output['categorical'] = {'indicators': None, 'values': []}
            if self.predict_indicators:
                # Indicator output shape: [batch_size, max_timeseries_length, n_categorical_feats]
                output['categorical']['indicators'] = self.categorical_indicator_mlp(indicator_embed)
            # Each head produces predictions for one categorical feature
            # Value output shape: [(batch_size, max_timeseries_length, n_classes) x n_categorical_feats]
            output['categorical']['values'] = [head(val_embed) for head in self.categorical_heads]

        if self.predict_text_feats:
            output['text'] = {'indicators': None, 'values': []}
            if self.predict_indicators:
                # Indicator output shape: [batch_size, max_timeseries_length, n_text_features]
                output['text']['indicators'] = self.text_indicator_mlp(indicator_embed)
            # Each head produces predictions for one text feature
            # Value output shape: [(batch_size, max_timeseries_length, TEXT_EMBED_DIM) x n_text_features]
            output['text']['embedded_values'] = [head(val_embed) for head in self.text_heads]

        # Add the timestamps and masks to the output (convenience for downstream discriminator)
        output['times'] = timestamps  # (batch_size, max_timeseries_length)
        output['masks'] = timestep_masks  # (batch_size, max_timeseries_length)

        return output


class MaskedTokenDiscriminator(torch.nn.Module):
    """A classifier that uses the output of a transformer encoder to make predictions.

    Attributes:
        encoder (nn.Module): The transformer encoder module that processes the input sequences.
        linear0 (torch.nn.Linear): The first linear layer in the feedforward network.
        linear (torch.nn.Linear): The final linear layer that maps to the number of classes.
    """

    def __init__(
        self, 
        encoder: torch.nn.Module,
        d_model: int,
        n_numeric_features: int,
        n_categorical_features: int,
        n_text_features: int = 0,
        n_static_features: int = 0,
        dim_feedforward: int = 64
    ):
        """Initialize an instance.

        Args:
            encoder (nn.Module): The transformer encoder module that processes the input sequences.
            d_model (int): The dimensionality of the model's embedding vectors.
            n_numeric_feats (int): The number of numeric features to predict
            n_categorical_feats (int): The number of categorical features to predict
            n_text_feats (int): The number of text features to predict. Defaults to zero.
            n_static_features (int): The number of static (non-time-varying) features to concatenate with the output
                from the encoder. Defaults to zero.
            dim_feedforward (int): The dimension of the feedforward layer.
        """

        super().__init__()

        self.encoder = encoder
        self.feedforward = torch.nn.Linear(d_model + n_static_features, dim_feedforward)

        if n_numeric_features > 0:
            self.numeric_head = torch.nn.Linear(dim_feedforward, n_numeric_features)
        else:
            self.numeric_head = None
        
        if n_categorical_features > 0:
            self.categorical_head = torch.nn.Linear(dim_feedforward, n_categorical_features)
        else:
            self.categorical_head = None

        if n_text_features > 0:
            self.text_head = torch.nn.Linear(dim_feedforward, n_text_features)
        else:
            self.text_head = None

        self.predict_numeric_feats = (self.numeric_head is not None)
        self.predict_categorical_feats = (self.categorical_head is not None)
        self.predict_text_feats = (self.text_head is not None)

    def forward(
        self, 
        batch: ValueAssociatedTensorData,
        static_data: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Perform a forward pass through the model.

        Args:
            batch (ValueAssociatedTensorData): A batch of data containing the input sequences and their associated 
                values generated by MaskedTokenGenerator.
            static_data (Tensor, optional): Static data (non- time-varying features) to be concatenated with the output 
                from the encoder.

        Returns:
            Dict[str, Tensor]: A dictionary containing predictions about whether or not the input was simulated for     
                each feature type.
        """
        # NOTE Xu et al. didn't use static data for pretraining in the final model. It's probably a good idea to use it, though, because static data may be important for learning the relationships between time-varying features. In fact, they should be used by the generator as well, and Xu et al. didn't allow for static features in their generator definition. I follow Xu et al. in not using static data to pretrain the models, but future studies should consider using the static data in the generator and discriminator. 

        inds_to_concat = []
        vals_to_concat = []

        if self.predict_numeric_feats:
            inds_to_concat.append(batch['numeric']['indicators'])
            vals_to_concat.append(torch.cat(batch['numeric']['values'], dim=2))

        if self.predict_categorical_feats:
            inds_to_concat.append(batch['categorical']['indicators'])
            vals_to_concat.append(torch.cat(batch['categorical']['values'], dim=2))

        if self.predict_text_feats:
            inds_to_concat.append(batch['text']['indicators'])
            if 'embedded_values' in batch['text']:
                text_embeddings = batch['text']['embedded_values'].flatten(start_dim=2)
                vals_to_concat.append(text_embeddings)
            else:
                raise ValueError("Expected 'embedded_values' key in batch['text'] for discriminator input.")

        # Concatenate the tensors for numeric and categorical features along the feature dimension
        #   val_indicators shape: (batch_size, max_timeseries_length, n_num_feats + n_cat_feats)
        #   val_values shape: (batch_size, max_timeseries_length, total_feat_dim)
        val_indicators = torch.cat(inds_to_concat, dim=2)
        val_values = torch.cat(vals_to_concat, dim=2)

        timestamps = batch['times']  # (batch_size, max_timeseries_length)
        timestep_masks = batch['masks']  # (batch_size, max_timeseries_length)

        # val_embed shape: (batch_size, max_timeseries_length, d_model)
        val_embed = self.encoder(
            val_indicators,  # (batch_size, max_timeseries_length, n_num_feats + n_cat_feats + n_text_feats)
            val_values,  # (batch_size, max_timeseries_length, total_feat_dim)
            timestamps,  # (batch_size, max_timeseries_length)
            timestep_masks  # (batch_size, max_timeseries_length)
        )
        val_embed = val_embed * timestep_masks.unsqueeze(-1)  # zero-out padding embeddings
        if static_data is not None:
            # Unsqueeze and expand the static data to match max_timeseries_length
            #   new static_data shape: (batch_size, max_timeseries_length, static_dim)
            static_data = static_data.unsqueeze(1).expand(-1, val_embed.size(1), -1)
            # Concatenate static data to the output embedding
            #   new val_embed shape: (batch_size, max_timeseries_length, total_feat_dim + static_dim)
            val_embed = torch.cat((val_embed, static_data), dim=2)
        val_embed = self.feedforward(val_embed)  # (batch_size, max_timeseries_length, dim_feedforward)
        val_embed = torch.nn.functional.gelu(val_embed)  # Apply GELU activation

        # Initialize the output structure for each feature type
        output = {}

        if self.predict_numeric_feats:
            output['numeric'] = self.numeric_head(val_embed)  # (batch_size, max_timeseries_length, n_num_feats)
        if self.predict_categorical_feats:
            output['categorical'] = self.categorical_head(val_embed)  # (batch_size, max_timeseries_length, n_cat_feats)
        if self.predict_text_feats:
            output['text'] = self.text_head(val_embed)  # (batch_size, max_timeseries_length, n_text_feats)

        return output


class TransformerHawkesProcess(torch.nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
        self,
        encoder: torch.nn.Module,
        num_types: int
    ):
        """Initialize an instance.
        
        Args:
            encoder (torch.nn.Module): The encoder module to use.
            num_types (int): The number of unique event types in the dataset.
        """

        super().__init__()
        self.encoder = encoder
        self.num_types = num_types
        # Initialize the alpha (decay) and beta (base intensity) parameters from equation (6) of Zuo et al. (2020).
        # Paper: https://arxiv.org/pdf/2002.09291.
        # See https://github.com/ant-research/EasyTemporalPointProcess
        # NOTE The original implementation by Zuo et al. did not define the loss function correctly, and the error 
        # carried through to Xu et al.'s codebase.
        self.intensity_base = torch.nn.Parameter(torch.empty([1, self.num_types]))
        self.intensity_decay = torch.nn.Parameter(torch.empty([1, self.num_types]))
        torch.nn.init.xavier_normal_(self.intensity_base)
        torch.nn.init.xavier_normal_(self.intensity_decay)

        # Bias is not used in the linear layer; it is handled separately as self.intensity_base.
        self.intensity_linear = torch.nn.Linear(self.encoder.d_model, self.num_types, bias=False)
        
        # Xu et al. did not use a bias term in the prediction layers
        # Given a history up to the (i-1)th time step, predict the (ith) event type and the time delta to that event
        self.time_predictor = torch.nn.Linear(self.encoder.d_model, 1, bias=False)
        self.type_predictor = torch.nn.Linear(self.encoder.d_model, self.num_types, bias=False)
        torch.nn.init.xavier_normal_(self.time_predictor.weight)
        torch.nn.init.xavier_normal_(self.type_predictor.weight)

        self.softplus = torch.nn.Softplus()
    
    def compute_initial_intensity(
        self,
        batch_size: int
    ) -> Tensor:
        """
        Compute initial intensity state for the first event in sequences.
        
        The initial intensity represents the model's prior belief about event likelihood
        before observing any events, based only on the base intensity parameters.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            Tensor: Initial intensity state of shape [batch_size, 1, n_event_types]
        """
        initial_intensity_state = self.intensity_base[None, ...].repeat(batch_size, 1, 1)
        return self.softplus(initial_intensity_state)

    def compute_conditional_intensity(
        self,
        encodings: Tensor,
        prev_event_times: Tensor,
        time_diff: Tensor
    ) -> Tensor:
        """Computes pre-softplus intensity states conditioned on event history.

        This method computes the pre-softplus intensity state for each event type at each time step in a timeseries of event encodings. The intensity state at each time step is conditioned on the encoding of the event history up to but not including that time step. Therefore, it is only defined for events occurring after the first observed event in the sequence.

        Because the conditional intensity is only defined for events after the first observed event, the tensors input as arguments to this function should be sequences one step shorter than the full event sequence. For `encodings` and `prev_event_times`, the last time step should be omitted. For `time_diff`, assuming that it was calculated using the `_calc_time_diff` method of the `TransformerHawkesLoss` class, the first time step (which is always zero) should be omitted.
        
        Args:
            model (TransformerHawkesProcess): An instance of the TransformerHawkesProcess model
            encodings (Tensor): A `Tensor` of shape [batch size, seq_len - 1, d_model] containing the hidden 
                representations of the event sequence from `model`'s encoder. The sequence length is reduced by 1 because the intensity is only conditioned on the history for events after the first one. 
            prev_event_times (Tensor): A `Tensor` of shape [batch size, seq_len - 1] containing the previous observed 
                event times for each step in the timeseries.
            time_diff (Tensor): A `Tensor` of shape [batch size, seq_len - 1] or 
                [batch size, seq_len - 1, n_samples] containing, respectively, the time differences between consecutive observed events or the time differences between sampled inter-event times and the previous observed event time.

        Returns:
            Tensor: A `Tensor` of shape [batch size, seq_len - 1, n_event_types] or 
                [batch size, seq_len - 1, n_samples, n_event_types] containing the pre-softplus intensity states for each event type at each time step.
        """
        eps = torch.finfo(torch.float32).eps
        
        if time_diff.dim() == 2:  # Observed event times
            decay = self.intensity_decay[None, ...]  # (1, 1, n_event_types)
            base_intensity = self.intensity_base[None, ...]  # (1, 1, n_event_types)
            
            prev_times = prev_event_times[..., None]  # (batch_size, seq_len - 1, 1)
            time_diff_expanded = time_diff[..., None]  # (batch_size, seq_len - 1, 1)

            current = decay * (time_diff_expanded / (prev_times + eps))
            history = self.intensity_linear(encodings)
            
            conditional_intensity_state = current + history + base_intensity
            
        elif time_diff.dim() == 3:  # Sampled inter-event times for Monte Carlo integration
            decay = self.intensity_decay[None, None, ...]  # (1, 1, 1, n_event_types)
            base_intensity = self.intensity_base[None, None, ...]  # (1, 1, 1, n_event_types)
            
            prev_times = prev_event_times[..., None, None]  # (batch_size, seq_len - 1, 1, 1)
            time_diff_expanded = time_diff[..., None]  # (batch_size, seq_len - 1, n_samples, 1)
            
            current = decay * (time_diff_expanded / (prev_times + eps))
            history = self.intensity_linear(encodings)[:, :, None, :]
            
            conditional_intensity_state = current + history + base_intensity

        else:

            raise ValueError(
                f"Unsupported time_diff shape: {time_diff.shape}. "
                f"Expected shape (batch_size, seq_len) or (batch_size, seq_len, n_samples)."
            )
        
        return self.softplus(conditional_intensity_state)

    def forward(self, batch: EventAssociatedTensorData) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate the hidden representations and predict the next event type and time.

        Hidden representations, next event, and time of next event are predicted for each token in the input sequence 
        of events. For a sequence of events (l_1, l_2, ..., l_N), the forward pass predicts (l_2, ..., l_N, l_{N+1}), where l_i yields a single predicted event type if the model was not initialized to produce multilabel predictions.

        Note that the predicted event types/times are not normalized, that is, they are real-valued outputs from the 
        linear layer.

        Args:
            batch (EventAssociatedTensorData): A batch of event-associated data containing the event indicators, event 
                timestamps, and timestep non-padding masks.
        
        Returns:
            Tuple(Tensor, Tuple(Tensor, Tensor)): A tuple containing the hidden representations of the input sequence
                (a `Tensor`), and a tuple containing the predicted next event type and time of the next event (two 
                `Tensor`s). The hidden representations are of shape [batch size, sequence length, d_model]. The event 
                type prediction is a `Tensor` with shape [batch size, sequence length, num_types], and the time 
                prediction is a `Tensor` with shape [batch size, sequence length, 1].
        """

        event_indicators = batch['indicators']  # [batch size, sequence length, num_types]
        event_times = batch['times']  # [batch size, sequence length]
        non_pad_mask = batch['masks']  # [batch size, sequence length]
        enc_output = self.encoder(event_indicators, event_times, non_pad_mask)  # [batch size, sequence length, d_model]
        
        # NOTE In the originally published code, the event types and time predictions were raw outputs from a linear 
        # layer, but this implementation applies a softplus activation to the time prediction to ensure it is 
        # non-negative.
        type_pred = self.type_predictor(enc_output)
        time_pred = self.softplus(self.time_predictor(enc_output))
        # Mask padding time steps in the predictions
        type_pred *= non_pad_mask[..., None]  # [batch_size, max_ts_len, n_types]
        time_pred *= non_pad_mask[..., None]  # [batch_size, max_ts_len, 1]

        return enc_output, (type_pred, time_pred)
