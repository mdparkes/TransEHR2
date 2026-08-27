"""Layers for Transformer Hawkes Process (THP)"""
import math
import torch

from torch import Tensor
from typing import Optional, Tuple

from TransEHR2.constants import PAD


def build_key_padding_attention_mask(non_padding_mask: Tensor) -> Tensor:
    r"""Turn a per-timestep padding mask into a self-attention mask.

    Args:
        non_padding_mask (Tensor): A [batch_size, seq_len] tensor that is 1/True at observed
            timesteps and 0/False at padding.

    Returns:
        Tensor: A boolean [batch_size, 1, seq_len] tensor that is True where the *key* position is
            padding, matching the polarity `MultiHeadAttention` expects (True means "do not
            attend"). Query rows are not masked here; padded query rows are zeroed by
            `EncoderLayer` via `non_padding_mask` instead.

    Note:
        The query axis is left at size 1 to broadcast rather than expanded to `seq_len`. Every
        query row of a bidirectional encoder masks the same keys, so the square form carries no
        extra information and would cost a [batch_size, seq_len, seq_len] bool -- 60 MB at batch
        200 and 550 timesteps -- once the attention inverts it. The event encoder's mask is
        genuinely two-dimensional because it also carries the causal triangle, and is built
        separately.
    """
    return non_padding_mask.eq(PAD).unsqueeze(1).to(torch.bool)


def build_frequency_ladder(n_bands: int, p_min: float, p_max: float) -> Tensor:
    r"""Build a ladder of band time-constants, spaced log-uniformly in period.

    .. math::
        P_m = P_{min} \left(\frac{P_{max}}{P_{min}}\right)^{(m-1)/(M-1)}, \quad
        \lambda_m = \frac{P_m}{2\pi}, \quad m = 1 \ldots M

    Args:
        n_bands (int): The number of bands, M.
        p_min (float): The shortest period, in the same units as the timestamps.
        p_max (float): The longest period, in the same units as the timestamps.

    Returns:
        Tensor: A [n_bands] tensor of :math:`\lambda`, ascending.

    Raises:
        ValueError: If `n_bands` is not positive, if `p_min` is not positive, or if `p_max` is not
            greater than `p_min`.

    Note:
        The buffer holds :math:`\lambda`, not the period, because every consumer computes
        `t / lambda`. A band carries usable information about a gap :math:`\Delta` only while its
        phase change is neither frozen nor wrapped, :math:`0.1 \le \Delta / \lambda \le \pi`,
        which in periods is :math:`2\Delta \le P \le 20\pi\Delta`. Hence the bounds this is
        normally called with: :math:`P_{min} = 2 \Delta_{min}` and
        :math:`P_{max} \approx 63 \Delta_{max}`. Each band then covers about 1.50 decades of
        :math:`\Delta`, so the count informative at any one gap size is
        :math:`M \cdot 1.50 / \log_{10}(P_{max} / P_{min})`.

        For contrast, the constant this replaces -- `exp(m log(1e4) / d_model)` -- is a
        :math:`\lambda` ladder running 1 to 9,330, so periods 6.28 h to 58,470 h. That range was
        chosen for token indices. Applied to a gap measured in hours it leaves *no* band coherent
        beyond about 10^5 h, which is inside the observed range of this dataset.
    """
    if n_bands < 1:
        raise ValueError(f'n_bands must be positive, got {n_bands}.')
    if p_min <= 0:
        raise ValueError(f'p_min must be positive, got {p_min}.')
    if p_max <= p_min:
        raise ValueError(f'p_max ({p_max}) must be greater than p_min ({p_min}).')

    steps = torch.arange(n_bands, dtype=torch.float64)
    # A one-band ladder has no range to span; it sits at p_min.
    exponent = steps / (n_bands - 1) if n_bands > 1 else steps
    periods = p_min * (p_max / p_min) ** exponent
    return (periods / (2.0 * math.pi)).to(torch.float32)


def get_activation_module(activation: str) -> torch.nn.Module:
    r"""Return the activation module named by `activation`.

    Args:
        activation (str): Either 'relu' or 'gelu'.

    Returns:
        torch.nn.Module: The corresponding activation module.

    Raises:
        ValueError: If `activation` is neither 'relu' nor 'gelu'.
    """
    if activation == 'relu':
        return torch.nn.ReLU()
    if activation == 'gelu':
        return torch.nn.GELU()
    raise ValueError(f'activation: expected "relu" or "gelu", got {activation}')


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
        normalize_before: bool = False,
        activation: str = 'gelu',
        query_key_transform: Optional[torch.nn.Module] = None
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
                feed-forward layers. If False, layer normalization is applied *after* each. Defaults to False.
            activation (str, optional): The activation used in the feed-forward network, either 'relu' or 'gelu'.
                Defaults to 'gelu', which is what this layer applied unconditionally before the argument existed.
            query_key_transform (torch.nn.Module, optional): A position-dependent transform applied to the query
                and key tensors after projection and head-splitting. See `MultiHeadAttention` for the contract.
                Defaults to None, which leaves the attention unchanged.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before,
            query_key_transform=query_key_transform)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before, activation=activation)

    def forward(
        self, 
        x: Tensor, 
        non_padding_mask: Optional[Tensor] = None,
        self_attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Performs a forward pass through the encoder layer.

        Args:
            x (Tensor): The input tensor to the encoder layer, [batch_size, seq_len, d_model].
            non_padding_mask (Tensor, optional): The mask tensor indicating non-padding positions, either
                [batch_size, seq_len] or [batch_size, seq_len, 1]. Padded positions are zeroed after
                attention and again after the feed-forward network. Defaults to None.
            self_attention_mask (Tensor, optional): A boolean mask that is True at key positions that must
                not be attended to, broadcastable to [batch_size, seq_len, seq_len]. A key-padding mask
                whose query axis is 1 broadcasts and costs nothing to invert; see
                `build_key_padding_attention_mask`. Defaults to None.
            positions (Tensor, optional): A [batch_size, seq_len] tensor of positions -- timestamps in this
                model -- forwarded to `query_key_transform`. Ignored when no transform is installed.
                Defaults to None.
            need_weights (bool, optional): Whether to return the explicit attention weight matrix. Materializing
                it costs a [batch_size, n_head, seq_len, seq_len] tensor, so it is off by default and the
                returned weights are then None. Defaults to False.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The output tensor from the encoder layer and, if `need_weights`,
                the self-attention weights.
        """

        x, attn = self.self_attention(
            x, x, x, mask=self_attention_mask, positions=positions, need_weights=need_weights
        )

        if non_padding_mask is not None:
            if non_padding_mask.dim() == 2:
                non_padding_mask = non_padding_mask.unsqueeze(-1)
            x = x * non_padding_mask

        x = self.pos_ffn(x)
        if non_padding_mask is not None:
            x = x * non_padding_mask

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
        normalize_before: bool = True,
        query_key_transform: Optional[torch.nn.Module] = None
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
            query_key_transform (torch.nn.Module, optional): A position-dependent transform applied to the
                query and key tensors *after* the W_q/W_k projections and the split into heads, and before the
                dot product. It is called as `query_key_transform(q, k, positions)` with `q` shaped
                [batch_size, n_head, len_q, d_k], `k` shaped [batch_size, n_head, len_k, d_k] and `positions`
                shaped [batch_size, seq_len], and must return a `(q, k)` pair of the same shapes. This is the
                seam a rotary encoding occupies: a rotation applied here makes the attention score a function
                of the position *difference*, which is not achievable by transforming the layer input. A shared
                instance may be passed to several layers, so a transform must be stateless across calls.
                Defaults to None, which leaves the attention unchanged.
        """

        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.query_key_transform = query_key_transform

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

    def forward(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Perform a forward pass through the multi-head attention layer.

        Args:
            q (Tensor): The query `Tensor` with shape [batch size, sequence length, d_model]
            k (Tensor): The key `Tensor` with shape [batch size, seuqnec length, d_model] 
            v (Tensor): The value `Tensor` with shape [batch size, sequence length, d_model]
            mask (Tensor, optional): Boolean mask `Tensor` preventing attention at the indicated positions
                in the input sequence. Defaults to None.
            positions (Tensor, optional): A [batch size, sequence length] tensor of positions passed to
                `self.query_key_transform`. Ignored when no transform is installed. Defaults to None.
            need_weights (bool, optional): Whether to return the explicit attention weight matrix. Defaults
                to False, in which case the returned weights are None.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The output `Tensor` and, if `need_weights`, the attention weights.
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

        # The rotary seam. q and k are past W_q/W_k and split into heads, which is the only place a
        # rotation by position yields a score that depends on the position difference alone.
        if self.query_key_transform is not None:
            q, k = self.query_key_transform(q, k, positions)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask, need_weights=need_weights)

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

    def __init__(
        self, 
        d_in: int, 
        d_hid: int, 
        dropout: float = 0.1, 
        normalize_before: bool = True, 
        activation: str = 'gelu'
    ):
        r"""Initialize an instance.
        
        Args:
            d_in (int): The input dimension of the feed-forward network.
            d_hid (int): The hidden dimension of the feed-forward network.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            normalize_before (bool, optional): Whether to apply layer normalization before the feed-forward network. If
                False, layer normalization is applied *after* the feed-forward network. Defaults to True.
            activation (str, optional): The activation applied after the first linear layer, either 'relu' or
                'gelu'. Defaults to 'gelu', which is what this layer applied unconditionally before the argument
                existed.
        """
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = torch.nn.Linear(d_in, d_hid)
        self.w_2 = torch.nn.Linear(d_hid, d_in)
        self.activation = get_activation_module(activation)

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

        x = self.activation(self.w_1(x))
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
        self.attn_dropout = attn_dropout
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        mask: Optional[Tensor] = None, 
        need_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Perform a forward pass through the layer
        
        Args:
            q (Tensor): The query tensor, [batch_size, n_head, len_q, d_k].
            k (Tensor): The key tensor, [batch_size, n_head, len_k, d_k].
            v (Tensor): The value tensor, [batch_size, n_head, len_k, d_v].
            mask (Tensor, optional): A boolean mask that is True at key positions that must not be attended
                to, broadcastable to [batch_size, n_head, len_q, len_k]. Defaults to None.
            need_weights (bool, optional): Whether to build the attention weight matrix explicitly and return
                it. Defaults to False.
        
        Returns:
            Tuple[Tensor, Optional[Tensor]]: The output tensor and, if `need_weights`, the attention weights.

        Note:
            The default path delegates to `torch.nn.functional.scaled_dot_product_attention`, which never
            materializes the [batch_size, n_head, len_q, len_k] score matrix. The `need_weights` path is the
            original explicit implementation, kept because it is the reference the fused path is checked
            against and because callers that want the weights have no other way to get them.

            The two paths agree to floating-point tolerance everywhere except at a query row whose keys are
            *all* masked. The explicit path fills those scores with -1e9, so the softmax comes back uniform
            and the row returns the mean of `v`; the fused path would return NaN, so such rows are given the
            zero vector instead. Every fully-masked row in this model is a padding row -- a padded timestep in
            the value encoder, or a leading padded timestep under the event encoder's causal mask -- and
            `EncoderLayer` multiplies padding rows by zero on the way out, so both paths deliver zero there.
            Zero, unlike NaN, survives that multiplication.
        """

        if need_weights:
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

            if mask is not None:
                attn = attn.masked_fill(mask, -1e9)

            attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
            output = torch.matmul(attn, v)

            return output, attn

        attend_mask, fully_masked = None, None
        if mask is not None:
            fully_masked = mask.all(dim=-1, keepdim=True)
            # scaled_dot_product_attention reads a boolean mask as True == "attend here", the opposite of
            # this layer's convention. Rows that are masked everywhere are released before inverting so the
            # softmax has something to normalize over, then zeroed afterwards.
            attend_mask = ~(mask & ~fully_masked)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attend_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=1.0 / self.temperature,
        )

        if fully_masked is not None:
            output = output.masked_fill(fully_masked, 0.0)

        return output, None


class TemporalPositionEncoding(torch.nn.Module):
    """Add a timestamp-dependent positional encoding to the embedded input tensor.

    Attributes:
        d_model (int): The dimension of embedded vectors in the input sequence.
        max_len (int): The maximimum length of the input sequence of embedded vectors.
        dropout (float): The dropout rate to apply to the output.
    """

    def __init__(
        self, 
        d_model: int, 
        dropout: Optional[float] = None,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        additive: bool = True
    ):
        """Initialize an instance.
        
        Args:
            d_model (int): The dimension of embedded vectors in the input sequence.
            dropout (float, optional): The dropout rate to apply to the output. Defaults to None.
            p_min (float, optional): The shortest band period, in the units of the timestamps. Must be
                given together with `p_max`. If both are None the ladder is the inherited
                `exp(m log(1e4) / d_model)` progression. Defaults to None.
            p_max (float, optional): The longest band period. Defaults to None.
            additive (bool, optional): Whether to add the encoding to the input. If False the layer keeps
                only its dropout and its padding mask, which is what the rotary arm needs: the rotation
                happens inside the attention, but the regularizer and the zeroing of padded positions have
                to stay, or the two arms differ in more than their encoding. Defaults to True.

        Raises:
            ValueError: If exactly one of `p_min` and `p_max` is given, or if the pair is invalid.
        """

        super().__init__()
        if (p_min is None) != (p_max is None):
            raise ValueError('p_min and p_max must be given together, or neither.')

        self.additive = additive
        if p_min is None:
            m = torch.arange(0, d_model, 2, dtype=torch.float32)
            position_encoding = torch.exp((m * math.log(1.e4) / d_model))
        else:
            position_encoding = build_frequency_ladder(d_model // 2, p_min, p_max)
        # Non-persistent: re-spanning the ladder changes its values but not its shape, so a persistent
        # buffer would let any checkpoint written before the change silently overwrite the new ladder with
        # the old one -- no error, and the arm running is not the arm intended.
        self.register_buffer('position_encoding', position_encoding, persistent=False)
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
        if self.additive:
            pos_enc = torch.zeros_like(x)
            scaled_times = torch.div(times, self.position_encoding)
            pos_enc[:, :, 0::2] = torch.sin(scaled_times)
            pos_enc[:, :, 1::2] = torch.cos(scaled_times)
            output = x + pos_enc
        else:
            output = x
        output = self.dropout(output) * non_padding_mask if self.dropout is not None else output * non_padding_mask
        return output


class RotaryTemporalEncoding(torch.nn.Module):
    r"""Continuous-position RoPE: rotate q and k by the record timestamp.

    Standard RoPE rotates by integer token index. This rotates by the timestamp itself, so the
    attention score between two records is a function of the elapsed time between them rather than
    of how many records separate them -- which is what an irregularly sampled series needs. It is a
    departure from the published method and is worth naming as one.

    Install it as a `query_key_transform` on `MultiHeadAttention`, whose contract it implements.

    Attributes:
        n_head (int): The number of attention heads.
        d_head (int): The per-head width of q and k. Must be even; each band occupies one channel pair.
        bands_per_head (int): `d_head // 2`.
        lambdas (Tensor): A [n_head, bands_per_head] non-persistent buffer of band time-constants.

    Note:
        **Bands are partitioned across heads, not repeated.** A rotation acts on two-dimensional
        blocks of the *per-head* q and k, so a single head holds only `d_head / 2` bands where the
        additive encoding holds `d_model / 2`. Giving every head the same ladder would leave the
        rotary arm with a two- to four-fold spectral deficit against the additive arm on the same
        gap range, and an additive win could then be read as a win for twice the bands rather than
        for the mechanism. Instead one ladder of `n_head * d_head / 2` bands is cut into contiguous
        per-head slices, head 0 taking the fastest. Each head resolves part of the timescale range
        rather than all of it -- specialization rather than a defect, and the same idea as ALiBi's
        per-head slopes, but it needs a sentence of justification in Methods.

        Parity with the additive arm therefore needs `n_head * d_head == d_model`. The value
        encoder gets that for free, since `d_head` is derived as `d_model // n_heads`. The event
        encoder does not: it takes `d_k` as its own config value, and `THP_ENCODER_D_K` was raised
        from 64 to 128 for exactly this reason. `d_v` stayed at 64 -- only q and k are rotated, so
        only their width sets the band count, and an asymmetric `d_k`/`d_v` is fine here because
        the softmax temperature is `d_k ** 0.5` either way. Lowering `d_k` again reopens the
        deficit silently, so check `band_count` against `d_model / 2` rather than assuming it.
    """

    def __init__(self, n_head: int, d_head: int, p_min: float, p_max: float):
        r"""Initialize an instance.

        Args:
            n_head (int): The number of attention heads.
            d_head (int): The per-head width of the query and key vectors, `d_k`.
            p_min (float): The shortest band period, in the units of the timestamps.
            p_max (float): The longest band period.

        Raises:
            ValueError: If `d_head` is odd, or if the ladder bounds are invalid.
        """
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(
                f'd_head must be even so that each band owns a channel pair, got {d_head}.'
            )
        self.n_head = n_head
        self.d_head = d_head
        self.bands_per_head = d_head // 2
        ladder = build_frequency_ladder(n_head * self.bands_per_head, p_min, p_max)
        # Non-persistent for the same reason as TemporalPositionEncoding's ladder: re-spanning
        # changes the values but not the shape, so a stale checkpoint would overwrite it in silence.
        self.register_buffer('lambdas', ladder.view(n_head, self.bands_per_head), persistent=False)

    @property
    def band_count(self) -> int:
        """The number of distinct frequencies in the ladder, for comparison against the additive arm."""
        return self.n_head * self.bands_per_head

    def forward(self, q: Tensor, k: Tensor, positions: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        r"""Rotate the query and key tensors by their record timestamps.

        Args:
            q (Tensor): Queries, [batch_size, n_head, seq_len, d_head].
            k (Tensor): Keys, [batch_size, n_head, seq_len, d_head].
            positions (Tensor): Record timestamps, [batch_size, seq_len].

        Returns:
            Tuple[Tensor, Tensor]: The rotated query and key tensors, same shapes.

        Raises:
            ValueError: If `positions` is None or does not match the sequence length of `q` and `k`.
        """
        if positions is None:
            raise ValueError(
                'RotaryTemporalEncoding needs the record timestamps; the encoder must pass '
                'positions through to the attention layer.'
            )
        if q.size(2) != positions.size(1) or k.size(2) != positions.size(1):
            raise ValueError(
                f'positions has sequence length {positions.size(1)}, but q and k have '
                f'{q.size(2)} and {k.size(2)}.'
            )

        cos, sin = self._sinusoids(positions, q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _sinusoids(self, positions: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        r"""Cosine and sine of `t / lambda`, [batch_size, n_head, seq_len, bands_per_head].

        Note:
            The angle is reduced modulo the period in float64 before the cosine and sine are taken.
            Against the fastest band of a five-decade ladder, `t / lambda` reaches about 4e5 radians
            at the far end of the history window, where float32 resolves a phase only to ~0.06 rad;
            reduced into [0, 2*pi) it is exact to 1e-7 instead. `torch.remainder` follows the sign of
            the divisor, so timestamps from before admission reduce correctly too. The float64
            intermediate is [batch_size, seq_len, n_head * bands_per_head] and is released as soon as
            the result is cast back down.
        """
        lambdas = self.lambdas.to(torch.float64)
        periods = lambdas * (2.0 * math.pi)
        times = positions.to(torch.float64)[:, None, :, None]
        angle = torch.remainder(times, periods[None, :, None, :]) / lambdas[None, :, None, :]
        angle = angle.to(dtype)
        return torch.cos(angle), torch.sin(angle)

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        r"""Turn each adjacent channel pair of `x` through the given angle.

        Note:
            Pairs are adjacent channels, (0, 1), (2, 3), ..., which matches the interleaving
            `TemporalPositionEncoding` already uses for its sine and cosine channels. The half-split
            convention some implementations use would be equally valid but would not line the two
            arms' ladders up channel for channel.
        """
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.flatten(start_dim=-2)


class TransformerBatchNormEncoderLayer(torch.nn.Module):
    r"""A custom implementation of TransformerEncoderLayer that uses BatchNorm instead of LayerNorm
    
    This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    The layer is batch-first unconditionally: it builds its `MultiheadAttention` with
    `batch_first=True` and expects `(batch_size, seq_len, d_model)` input. Unlike
    `torch.nn.TransformerEncoderLayer` it takes no `batch_first` argument, so callers that
    build either layer from a shared set of keyword arguments must not pass one.

    Note:
        `src_key_padding_mask` gates attention but not normalization: padded timesteps still
        contribute to the BatchNorm statistics. This matches the upstream mvts_transformer
        layer this derives from, and biases the running statistics toward whatever padding
        embeds to as the padded fraction of a batch grows.

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
        activation: str = "relu",
        norm_first: bool = False
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
            norm_first (bool, optional): Whether to apply normalization before attention and feedforward operations
                (Pre-LN). If False, normalization is applied after (Post-LN). Defaults to False.
        """

        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.norm_first = norm_first
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
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False
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
            src (Tensor): the sequence to the encoder layer, shaped
                `[batch_size, seq_len, d_model]`.
            src_mask (Tensor, optional): the mask for the src sequence.
            src_key_padding_mask (Tensor, optional): the mask for the src keys per batch.
            is_causal (bool, optional): a hint that `src_mask` is the causal mask, forwarded to
                the attention. `torch.nn.TransformerEncoder` passes this to every layer it calls,
                so the parameter has to be accepted. Defaults to False.
        Returns:
            Tensor: The result of a forward pass through the self-attention and feedforward layers. The shape of the
                output matches the input, `[batch_size, seq_len, d_model]`, where `seq_len` is the length of the input
                token sequence and `d_model` is the model's embedding dimensionality.
        """
        # BatchNorm1d(d_model) needs the feature axis second. permute(1, 2, 0) moves the last
        # axis into that slot and permute(2, 0, 1) is its exact inverse, so the pair round-trips
        # whatever layout it is given. BatchNorm pools over both non-channel axes, making the
        # statistics one mean/variance per feature over every (episode, timestep) position --
        # the same element set either way. Both are therefore still correct now that the encoder
        # feeds (batch_size, seq_len, d_model) rather than the transposed layout these permutes
        # were originally written for.
        if self.norm_first:
            # Pre-LN: normalize before attention and feedforward
            # Self-attention with pre-normalization
            src_normalized = src.permute(1, 2, 0)  # BatchNorm wants channels second, (seq_len, d_model, batch_size)
            src_normalized = self.norm1(src_normalized)
            src_normalized = src_normalized.permute(2, 0, 1)  # Restore, (batch_size, seq_len, d_model)
            src2 = self.self_attn(src_normalized, src_normalized, src_normalized, 
                                  attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                  is_causal=is_causal)[0]
            src = src + self.dropout1(src2)  # Residual connection
            # Feedforward with pre-normalization
            src_normalized = src.permute(1, 2, 0)  # BatchNorm wants channels second, (seq_len, d_model, batch_size)
            src_normalized = self.norm2(src_normalized)
            src_normalized = src_normalized.permute(2, 0, 1)  # Restore, (batch_size, seq_len, d_model)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src_normalized))))
            src = src + self.dropout2(src2)  # Residual connection
        else:
            # Post-LN: normalize after attention and feedforward (original behaviour)
            # Self-attention
            src2 = self.self_attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                is_causal=is_causal
            )[0]
            src = src + self.dropout1(src2)  # Residual connection, (batch_size, seq_len, d_model)
            src = src.permute(1, 2, 0)  # Reshape for BatchNorm, (seq_len, d_model, batch_size)
            src = self.norm1(src)  # Perform batch normalization
            src = src.permute(2, 0, 1)  # Restore original shape, (batch_size, seq_len, d_model)
            # Feedforward network
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)  # (batch_size, seq_len, d_model)
            src = src.permute(1, 2, 0)  # Reshape for BatchNorm, (seq_len, d_model, batch_size)
            src = self.norm2(src)  # Perform batch normalization
            src = src.permute(2, 0, 1)  # Restore original shape, (batch_size, seq_len, d_model)
        return src
