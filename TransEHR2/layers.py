"""Layers for Transformer Hawkes Process (THP)"""
import math
import torch

from torch import Tensor
from typing import Optional, Tuple

from TransEHR2.constants import PAD


class EncoderLayer(torch.nn.Module):
    r"""
    Encoder layer composed of a multi-head attention mechanism and a position-wise feed-forward network.

    Attributes:
        slf_attn (MultiHeadAttention): The multi-head attention mechanism.
        pos_ffn (PositionwiseFeedForward): The position-wise feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int, 
        n_head: int, 
        d_k: int, 
        d_v: int, 
        dropout: float = 0.1, 
        normalize_before: bool = True
    ):
        r"""Initialize an instance
        
        Args:
            d_model (int): The input/output dimension of the model.
            d_inner (int): The inner dimension of the feed-forward network.
            n_head (int): The number of attention heads.
            d_k (int): The dimension of the key vectors.
            d_v (int): The dimension of the value vectors.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            normalize_before (bool, optional): Whether to apply layer normalization before the attention and
                feed-forward layers. If False, layer normalization is applied *after* each. Defaults to True.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(
        self, 
        x: Tensor, 
        non_padding_mask: Optional[Tensor] = None,
        self_attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Performs a forward pass through the encoder layer.

        Args:
            enc_input (Tensor): The input tensor to the encoder layer.
            non_pad_mask (Tensor, optional): The mask tensor indicating non-padding positions. Defaults to None.
            slf_attn_mask (Tensor, optional): The mask tensor for self-attention. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The output tensor from the encoder layer and the self-attention weights.
        """

        x, attn = self.self_attention(x, x, x, mask=self_attention_mask)

        if non_padding_mask.dim() == 2:
            non_padding_mask = non_padding_mask.unsqueeze(-1)

        if non_padding_mask is not None:
            x *= non_padding_mask

        x = self.pos_ffn(x)
        if non_padding_mask is not None:
            x *= non_padding_mask

        return x, attn


class MultiHeadAttention(torch.nn.Module):
    r""" Multi-head attention layer
    
    Attributes:
        n_head (int): The number of attention heads.
        d_k (int): The dimension of the key and query vectors.
        d_v (int): The dimension of the value vectors.
        dropout (float): The dropout rate to apply throughout the layer. Defaults to 0.1.
        normalize_before (bool): Whether to apply layer normalization before the attention and feed-forward layers. If
            False, layer normalization is applied *after* each. Defaults to True.
    """

    def __init__(
        self, 
        n_head: int, 
        d_model: int, 
        d_k: int,
        d_v: int, 
        dropout: float = 0.1, 
        normalize_before: bool = True
    ):
        r"""Initialize an instance.

        Args:
            n_head (int): Number of attention heads.
            d_model (int): Dimension of the layer's inputs and outputs.
            d_k (int): Dimension of the key and query vectors.
            d_v (int): Dimension of the value vectors.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            normalize_before (bool, optional): Whether to apply layer normalization before self-attention. If
                False, layer normalization is applied after residual self-attention. Defaults to True.
        """

        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = torch.nn.Linear(d_v * n_head, d_model)
        torch.nn.init.xavier_uniform_(self.w_qs.weight)
        torch.nn.init.xavier_uniform_(self.w_ks.weight)
        torch.nn.init.xavier_uniform_(self.w_vs.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Perform a forward pass through the multi-head attention layer.

        Args:
            q (Tensor): The query `Tensor` with shape [batch size, sequence length, d_model]
            k (Tensor): The key `Tensor` with shape [batch size, seuqnec length, d_model] 
            v (Tensor): The value `Tensor` with shape [batch size, sequence length, d_model]
            mask (Tensor, optional): Boolean mask `Tensor` preventing attention at the indicated positions
                in the input sequence. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The output `Tensor` and the attention weights.
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: batch size x seq length x (n heads * dim)
        # Separate different heads: batch size x sequence length x n heads x dim
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose to batch size x n heads x seq length x dim for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(torch.nn.Module):
    r""" Two-layer position-wise feed-forward neural network.
    
    Attributes:
        normalize_before (bool): Whether to apply layer normalization before the feed-forward network. If
            False, layer normalization is applied *after* the feed-forward network. Defaults to True.
        w_1 (torch.nn.Linear): The first linear transformation.
        w_2 (torch.nn.Linear): The second linear transformation.
        layer_norm (torch.nn.LayerNorm): The LayerNorm layer.
        dropout (torch.nn.Dropout): The dropout layer.
    """

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1, normalize_before: bool = True):
        r"""Initialize an instance.
        
        Args:
            d_in (int): The input dimension of the feed-forward network.
            d_hid (int): The hidden dimension of the feed-forward network.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            normalize_before (bool, optional): Whether to apply layer normalization before the feed-forward network. If
                False, layer normalization is applied *after* the feed-forward network. Defaults to True.
        """
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = torch.nn.Linear(d_in, d_hid)
        self.w_2 = torch.nn.Linear(d_hid, d_in)

        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        r"""Perform a forward pass through the feed-forward network.
        
        Args:
            x (Tensor): The input tensor to the feed-forward network.
        
        Returns:
            Tensor: The output tensor from the feed-forward network.
        """

        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = torch.nn.functional.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class RNN_layers(torch.nn.Module):
    r"""
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = torch.nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = torch.nn.Linear(d_rnn, d_model)


    def forward(self, data, non_pad_mask):
        max_seq_len = non_pad_mask.size(1)
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = torch.nn.utils.rnn.pack_padded_sequence(
            data, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        print(f'RNN_layers // pack_enc_output.shape: {pack_enc_output.data.shape}')
        temp = self.rnn(pack_enc_output)[0]
        print(f'RNN_layers // temp.shape: {temp.data.shape}')
        out = torch.nn.utils.rnn.pad_packed_sequence(
            temp, padding_value=PAD, total_length=max_seq_len, batch_first=True
        )
        out = out[0]
        print(f'RNN_layers // out.shape: {out.data.shape}')
        out = self.projection(out)
        return out


class ScaledDotProductAttention(torch.nn.Module):
    """Perform scaled dot product attention
    
    Attributes:
        temperature (float): The temperature used to scale the dot product.
        dropout (nn.Dropout): The dropout layer.
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.2):
        """Initialize an instance
        
        Args:
            temperature (float): The temperature used to scale the dot product.
            attn_dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()

        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Perform a forward pass through the layer
        
        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.
            v (Tensor): The value tensor.
            mask (Tensor, optional): The mask tensor. Defaults to None.
        
        Returns:
            Tuple[Tensor, Tensor]: The output tensor and the attention weights.
        """

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class TemporalPositionEncoding(torch.nn.Module):
    """Add a timestamp-dependent positional encoding to the embedded input tensor.

    Attributes:
        d_model (int): The dimension of embedded vectors in the input sequence.
        max_len (int): The maximimum length of the input sequence of embedded vectors.
        dropout (float): The dropout rate to apply to the output.
    """

    def __init__(self, d_model: int, dropout: Optional[float] = None):
        """Initialize an instance.
        
        Args:
            d_model (int): The dimension of embedded vectors in the input sequence.
            dropout (float, optional): The dropout rate to apply to the output. Defaults to None.
        """

        super().__init__()
        m = torch.arange(0, d_model, 2, dtype=torch.float32)
        position_encoding = torch.exp((m * math.log(1.e4) / d_model))
        self.register_buffer('position_encoding', position_encoding)  # Store as non-trainable tensor the state dict.
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None


    def forward(self, x: Tensor, times: Tensor, non_padding_mask: Tensor) -> Tensor:
        """Calculate the time-dependent positional encodings of the input tensor.
        
        Args:
            x (Tensor): A sequence of embedded inputs, size [batch_size, max_seq_len, self.d_model].
            times (Tensor): A sequence of timestamps, size [batch_size, max_seq_len].
            non_padding_mask (Tensor): A mask tensor that is 1 or True at positions in the input sequence that
                are not padding, size [batch_size, max_seq_len].
        
        Returns:
            Tensor: The time-dependent positional embeddings of the input tensor with dropout applied.
        """
        if times.dim() == 2:
            times = times.unsqueeze(-1)
        if non_padding_mask.dim() == 2:
            non_padding_mask = non_padding_mask.unsqueeze(-1)
        x[:, :, 0::2] = x[:, :, 0::2] + torch.sin(torch.div(times, self.position_encoding))
        x[:, :, 1::2] = x[:, :, 1::2] + torch.cos(torch.div(times, self.position_encoding))
        x = self.dropout(x) * non_padding_mask if self.dropout is not None else x * non_padding_mask
        return x


class TransformerBatchNormEncoderLayer(torch.nn.Module):
    r"""A custom implementation of TransformerEncoderLayer that uses BatchNorm instead of LayerNorm
    
    This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Attributes:
        self_attn (torch.nn.modules.MultiheadAttention): A multi-head attention mechanism
        linear1 (torch.nn.modules.Linear): The first linear transformation in the feedforward network
        dropout (torch.nn.modules.Dropout): A dropout layer.
        linear2 (torch.nn.modules.Linear): The second linear transformation in the feedforward network
        norm1 (torch.nn.modules.BatchNorm1d): Batch normalization layer applied after the self-attention mechanism
        norm2 (torch.nn.modules.BatchNorm1d): Batch normalization layer applied after the feedforward network
        dropout1 (torch.nn.modules.Dropout): A dropout layer applied after the self-attention mechanisms
        dropout2 (torch.nn.modules.Dropout): A dropout layer applied after the feedforward network
        activation (torch.nn.ReLU | torch.nn.GELU): The activation applied after the first layer in the feedforward
            network
    """

    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: str = "relu"
    ):
        r"""Initialize an instance
        Args:
            d_model (int): The number of expected features (embedding dimension) for each item in the input sequence.
            n_heads (int): The number of attention heads.
            dim_feedforward (int, optional): The size of the hidden layers in the feedfoward network. Defaults to 2048.
            dropout (float, optional): The dropout rate applied after self-attention and the feedforward network.
                Defaults to 0.1.
            activation (str, optional): The activation applied after the first layer in the feedforward network. Accepts
                either "relu" or "gelu".
        """

        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Feedforward layers
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.BatchNorm1d(d_model, eps=1e-5)  # Normalize each feature across batch samples, timesteps
        self.norm2 = torch.nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        else:
            raise ValueError(f'activation: expected "relu" or "gelu", got {activation}')

    def __setstate__(self, state: dict) -> None:
        r"""Restore the state of the TransformerBatchNormEncoderLayer instance.

        This method is used during unpickling to restore the state of the object.
        If the 'activation' key is not present in the state dictionary, it sets
        the default activation function to ReLU.

        Args:
            state (dict): The state dictionary containing the attributes to be restored.
        """
        if 'activation' not in state:
            state['activation'] = torch.nn.ReLU()
        super().__setstate__(state)

    def forward(
        self, 
        src: Tensor, 
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        When performing multi-head self attention with a forward pass through `self.self_attn` (an instance of 
        torch.nn.modules.MultiheadAttention), the pass' `query`, `key`, and `value` positional parameters all take the
        same input sequence as their argument. The `attn_mask` parameter takes `src_mask` as its argument, which is a 
        `Tensor` mask that prevents attention at the indicated positions. The shape should be
        [target sequence length, source sequence length] or 
        [batch size * heads, target sequence length, source sequence length] if a different mask is to be used for each
        batch. The `key_padding_mask` parameter takes `src_key_padding_mask` as its argument, which is a `Tensor`
        indicating which elements of `key` should be ignored by the attention heads. One would pad an input sequence if
        it is shorter than a predefined length, and the padding mask would indicate which elements are padding.

        Args:
            src (Tensor): the sequence to the encoder layer.
            src_mask (Tensor, optional): the mask for the src sequence.
            src_key_padding_mask (Tensor, optional): the mask for the src keys per batch.
        Returns:
            Tensor: The result of a forward pass through the self-attention and feedforward layers. The shape of the
                output is `[seq_len, batch_size, d_model]`, where `seq_len` is the length of the input token sequence
                and `d_model` is the model's embedding dimensionality.
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # Residual connection, (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # Reshape for BatchNorm, (batch_size, d_model, seq_len)
        src = self.norm1(src)  # Perform batch normalization
        src = src.permute(2, 0, 1)  # Restore original shape, (seq_len, batch_size, d_model)
        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # Reshape for BatchNorm, (batch_size, d_model, seq_len)
        src = self.norm2(src)  # Perform batch normalization
        src = src.permute(2, 0, 1)  # Restore original shape, (seq_len, batch_size, d_model)
        return src
