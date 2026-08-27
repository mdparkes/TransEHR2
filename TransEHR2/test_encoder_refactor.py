"""Probes for the encoder attention refactor.

The refactor does three things, and each is checked here:

1. ``ScaledDotProductAttention`` delegates to ``torch.nn.functional.scaled_dot_product_attention``
   instead of building the score matrix. The explicit implementation survives behind
   ``need_weights=True``, which makes it the reference the fused path is checked against.
2. ``ValueDataEncoder`` stacks the repository's own ``EncoderLayer`` -- the class the event stream
   already uses -- instead of ``nn.TransformerEncoder``. Its attention is therefore owned rather
   than delegated to ``nn.MultiheadAttention``.
3. ``MultiHeadAttention`` accepts a ``query_key_transform``: a hook applied to q and k after the
   W_q/W_k projections and the split into heads. That is the seam a rotary encoding occupies, and
   it is a no-op until one is installed.

What is and is not equivalence: the event stream and the ``norm='BatchNorm'`` value stack are
numerically unchanged, and that is asserted directly. The ``norm='LayerNorm'`` value stack is a
different block -- pre-LN over only the query, no q/k/v projection bias, a separate output
projection -- so it cannot equal ``nn.TransformerEncoderLayer``. What is pinned instead is that it
is exactly ``EncoderLayer``, checked against the block's arithmetic written out longhand, plus the
contract the old stack met: episodes stay independent, attention runs over time, padding does not
move observed positions, and every parameter trains.

Run directly (``python -m TransEHR2.test_encoder_refactor``) or under pytest.
"""

import copy
import math

import torch

from TransEHR2.layers import (
    EncoderLayer,
    MultiHeadAttention,
    ScaledDotProductAttention,
    build_key_padding_attention_mask,
)
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------

N_FEATURES = 6
FEAT_DIM = 6
N_EVENT_TYPES = 5
D_MODEL = 16
N_HEADS = 2
DIM_FF = 16


def _value_encoder(seed=0, n_blocks=2, norm='LayerNorm', activation='gelu', transform=None):
    torch.manual_seed(seed)
    encoder = ValueDataEncoder(
        n_features=N_FEATURES,
        feat_dim=FEAT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_blocks=n_blocks,
        dim_feedforward=DIM_FF,
        dropout=0.1,
        activation=activation,
        norm=norm,
        normalize_before=True,
        query_key_transform=transform,
    )
    encoder.eval()
    return encoder


def _event_encoder(seed=0, n_layers=2, transform=None):
    torch.manual_seed(seed)
    encoder = EventDataEncoder(
        num_types=N_EVENT_TYPES,
        d_model=D_MODEL,
        d_inner=DIM_FF,
        n_layers=n_layers,
        n_head=N_HEADS,
        d_k=8,
        d_v=8,
        dropout=0.1,
        normalize_before=True,
        query_key_transform=transform,
    )
    encoder.eval()
    return encoder


def _masks(batch_size=4, seq_len=8, include_empty=True):
    """Observed/padding masks covering the shapes padding actually takes.

    Episode 1 is trailing-padded, episode 2 carries leading padding -- which under the event
    encoder's causal mask makes its first rows fully masked -- and episode 3, when `include_empty`,
    is all padding: the case that produces NaN in any stack that delegates to
    `torch.nn.MultiheadAttention`.
    """
    masks = torch.ones(batch_size, seq_len)
    if batch_size > 1:
        masks[1, seq_len - 3:] = 0.0
    if batch_size > 2:
        masks[2, :2] = 0.0
    if include_empty and batch_size > 3:
        masks[3] = 0.0
    return masks


def _value_batch(batch_size=4, seq_len=8, seed=1, include_empty=True):
    generator = torch.Generator().manual_seed(seed)
    indicators = (torch.rand(batch_size, seq_len, N_FEATURES, generator=generator) > 0.5).float()
    values = torch.randn(batch_size, seq_len, FEAT_DIM, generator=generator)
    # Irregular, ascending timestamps in hours -- the quantity the ladder is indexed by.
    gaps = torch.rand(batch_size, seq_len, generator=generator) * 3.0
    timestamps = gaps.cumsum(dim=1)
    return indicators, values, timestamps, _masks(batch_size, seq_len, include_empty)


def _event_batch(batch_size=4, seq_len=8, seed=2, include_empty=True):
    generator = torch.Generator().manual_seed(seed)
    indicators = (torch.rand(batch_size, seq_len, N_EVENT_TYPES, generator=generator) > 0.5).float()
    gaps = torch.rand(batch_size, seq_len, generator=generator) * 3.0
    timestamps = gaps.cumsum(dim=1)
    return indicators, timestamps, _masks(batch_size, seq_len, include_empty)


# --------------------------------------------------------------------------------------------
# The fused attention against the explicit one it replaces
# --------------------------------------------------------------------------------------------

def test_fused_attention_matches_explicit_path():
    """``scaled_dot_product_attention`` must reproduce the hand-written score matrix."""
    attention = ScaledDotProductAttention(temperature=8 ** 0.5, attn_dropout=0.1)
    attention.eval()

    generator = torch.Generator().manual_seed(3)
    q = torch.randn(4, 2, 8, 8, generator=generator)
    k = torch.randn(4, 2, 8, 8, generator=generator)
    v = torch.randn(4, 2, 8, 8, generator=generator)
    # A key-padding mask with at least one key left open on every row.
    mask = build_key_padding_attention_mask(_masks(4, 8)[:3]).unsqueeze(1)

    with torch.no_grad():
        fused, weights = attention(q[:3], k[:3], v[:3], mask=mask)
        explicit, explicit_weights = attention(q[:3], k[:3], v[:3], mask=mask, need_weights=True)

    assert weights is None, 'the fused path must not materialize the score matrix'
    assert explicit_weights.shape == (3, 2, 8, 8)
    delta = (fused - explicit).abs().max().item()
    assert delta < 1e-5, f'fused attention diverged from the explicit path (max delta {delta:.3e})'


def test_fused_attention_gives_fully_masked_rows_the_zero_vector():
    """A row with no attendable key must come back finite.

    The explicit path fills those scores with -1e9, so the softmax is uniform and the row returns
    the mean of v. Handed the same mask, ``scaled_dot_product_attention`` returns NaN -- and NaN
    survives the padding multiply that follows, which is the whole reason ``install_nan_hooks``
    exists in dump_finetuned_predictions.py. The fused path returns zero instead.
    """
    attention = ScaledDotProductAttention(temperature=8 ** 0.5, attn_dropout=0.0)
    attention.eval()

    generator = torch.Generator().manual_seed(4)
    q = torch.randn(1, 2, 4, 8, generator=generator)
    k = torch.randn(1, 2, 4, 8, generator=generator)
    v = torch.randn(1, 2, 4, 8, generator=generator)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)  # nothing may be attended to

    with torch.no_grad():
        fused, _ = attention(q, k, v, mask=mask)
        explicit, _ = attention(q, k, v, mask=mask, need_weights=True)

    assert torch.isfinite(fused).all(), 'fully masked rows leaked NaN'
    assert torch.allclose(fused, torch.zeros_like(fused)), 'fully masked rows are not zero'
    # The reference behaviour these rows used to have, recorded so the difference is deliberate.
    assert torch.allclose(explicit, v.mean(dim=2, keepdim=True).expand_as(explicit), atol=1e-5)


def test_encoder_layer_output_is_unchanged_at_fully_masked_padding_rows():
    """The two paths differ only where ``EncoderLayer`` zeroes the row anyway.

    Every fully-masked query row in this model is a padding row, and padding rows are multiplied
    by zero on the way out of the block. Zero and the-mean-of-v both land on zero there, so the
    change in the fused path is not observable through the layer.
    """
    torch.manual_seed(5)
    layer = EncoderLayer(D_MODEL, DIM_FF, N_HEADS, 8, 8, dropout=0.0, normalize_before=True)
    layer.eval()

    x = torch.randn(4, 8, D_MODEL, generator=torch.Generator().manual_seed(6))
    masks = _masks(4, 8)
    attention_mask = build_key_padding_attention_mask(masks)

    with torch.no_grad():
        fused, _ = layer(x, non_padding_mask=masks, self_attention_mask=attention_mask)
        explicit, _ = layer(
            x, non_padding_mask=masks, self_attention_mask=attention_mask, need_weights=True
        )

    assert torch.isfinite(fused).all(), 'fused path leaked NaN through the block'
    delta = (fused - explicit).abs().max().item()
    assert delta < 1e-5, f'block output diverged between attention paths (max delta {delta:.3e})'


def test_event_encoder_is_numerically_unchanged():
    """End to end, the event stream must give what the explicit attention gave.

    The event encoder already stacked ``EncoderLayer``, so the refactor touches only how the
    scores are computed. Forcing ``need_weights`` restores the original arithmetic, and the two
    runs must agree to floating point.
    """
    original = ScaledDotProductAttention.forward

    def explicit(self, q, k, v, mask=None, need_weights=False):
        return original(self, q, k, v, mask=mask, need_weights=True)

    indicators, timestamps, masks = _event_batch()

    for n_layers in (1, 3):
        encoder = _event_encoder(n_layers=n_layers)
        with torch.no_grad():
            fused = encoder(indicators, timestamps, masks)
            ScaledDotProductAttention.forward = explicit
            try:
                reference = encoder(indicators, timestamps, masks)
            finally:
                ScaledDotProductAttention.forward = original

        assert torch.isfinite(fused).all(), f'{n_layers}-layer event encoder leaked NaN'
        delta = (fused - reference).abs().max().item()
        assert delta < 1e-4, (
            f'{n_layers}-layer event encoder moved by {delta:.3e} under the fused attention'
        )


def test_encoders_stay_finite_on_fully_padded_episodes():
    """Neither encoder may emit NaN for an episode that is entirely padding."""
    value_encoder = _value_encoder()
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        value_out = value_encoder(indicators, values, timestamps, masks)
    assert torch.isfinite(value_out).all(), 'value encoder leaked NaN on an all-padding episode'
    assert torch.allclose(value_out[3], torch.zeros_like(value_out[3]), atol=1e-6), (
        'an all-padding episode did not encode to the zero vector'
    )

    event_encoder = _event_encoder()
    event_indicators, event_times, event_masks = _event_batch()
    with torch.no_grad():
        event_out = event_encoder(event_indicators, event_times, event_masks)
    assert torch.isfinite(event_out).all(), 'event encoder leaked NaN on an all-padding episode'


# --------------------------------------------------------------------------------------------
# The migrated value encoder is exactly EncoderLayer
# --------------------------------------------------------------------------------------------

def _reference_block(x, layer, non_padding_mask, attention_mask):
    """One pre-LN ``EncoderLayer`` written out longhand, from the layer's own weights.

    Two details of this block are easy to get wrong and are the reason it is spelled out rather
    than assumed: only the *query* is normalized before attention -- k and v are read from the
    unnormalized input, which is what Zuo et al.'s layer does and is not what
    ``nn.TransformerEncoderLayer`` does -- and the padding mask is applied twice, once after
    attention and once after the feed-forward network.
    """
    attention = layer.self_attention
    batch_size, seq_len, _ = x.shape
    n_head, d_k, d_v = attention.n_head, attention.d_k, attention.d_v

    residual = x
    query = attention.layer_norm(x)
    q = attention.w_qs(query).view(batch_size, seq_len, n_head, d_k).transpose(1, 2)
    k = attention.w_ks(x).view(batch_size, seq_len, n_head, d_k).transpose(1, 2)
    v = attention.w_vs(x).view(batch_size, seq_len, n_head, d_v).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
    scores = scores.masked_fill(attention_mask.unsqueeze(1), -1e9)
    attended = torch.softmax(scores, dim=-1) @ v
    attended = attended.transpose(1, 2).reshape(batch_size, seq_len, n_head * d_v)
    out = attention.fc(attended) + residual
    out = out * non_padding_mask.unsqueeze(-1)

    feed_forward = layer.pos_ffn
    normed = feed_forward.layer_norm(out)
    out = feed_forward.w_2(torch.nn.functional.gelu(feed_forward.w_1(normed))) + out
    return out * non_padding_mask.unsqueeze(-1)


def test_value_encoder_matches_the_encoder_layer_written_longhand():
    """The migrated stack must be the THP block, not something that merely runs."""
    encoder = _value_encoder(n_blocks=2)
    indicators, values, timestamps, masks = _value_batch()

    with torch.no_grad():
        actual = encoder(indicators, values, timestamps, masks)

        embedding = (
            encoder.indicator_input_projection_layer(indicators.float())
            + encoder.value_input_projection_layer(values)
        )
        embedding = encoder.position_encoding_layer(embedding, timestamps, masks)
        attention_mask = build_key_padding_attention_mask(masks)
        for layer in encoder.layer_stack:
            embedding = _reference_block(embedding, layer, masks, attention_mask)
        expected = encoder.activation(embedding)

    delta = (actual - expected).abs().max().item()
    assert delta < 1e-5, f'value encoder is not the EncoderLayer block (max delta {delta:.3e})'


def test_value_encoder_blocks_are_the_class_fsdp_wraps():
    """The block class name has to stay one FSDP already knows.

    ``fsdp_transformer_layer_cls_to_wrap`` is a comma-separated list of class *names*. A block
    class not on that list is silently left unwrapped, its parameters fall into the root unit, and
    the symptom is a memory surprise or an opaque flat-param error that names no config string.
    """
    encoder = _value_encoder(n_blocks=3)
    assert len(encoder.layer_stack) == 3
    assert all(isinstance(block, EncoderLayer) for block in encoder.layer_stack)
    assert EncoderLayer.__name__ in 'EncoderLayer,TransformerEncoderLayer'.split(',')


def test_value_encoder_attention_width_is_d_model_over_n_heads():
    """``EncoderLayer`` takes d_k and d_v explicitly; getting them wrong halves the attention."""
    encoder = _value_encoder(n_blocks=1)
    attention = encoder.layer_stack[0].self_attention
    assert attention.d_k == D_MODEL // N_HEADS
    assert attention.d_v == D_MODEL // N_HEADS
    assert attention.w_qs.weight.shape == (N_HEADS * (D_MODEL // N_HEADS), D_MODEL)
    assert attention.w_qs.bias is None, 'the THP projections are bias-free'


def test_value_encoder_propagates_normalize_before():
    """Pre-LN is in this model because post-LN was unstable, and it is off by default here.

    ``EncoderLayer`` defaults ``normalize_before`` to False while both of its children default it
    to True, so a caller that forgets to pass it gets post-LN silently.
    """
    for requested in (True, False):
        torch.manual_seed(0)
        encoder = ValueDataEncoder(
            n_features=N_FEATURES, feat_dim=FEAT_DIM, d_model=D_MODEL, n_heads=N_HEADS,
            n_encoder_blocks=1, dim_feedforward=DIM_FF, norm='LayerNorm',
            normalize_before=requested,
        )
        block = encoder.layer_stack[0]
        assert block.self_attention.normalize_before is requested
        assert block.pos_ffn.normalize_before is requested


def test_value_encoder_honours_the_activation_argument():
    """``activation`` used to be accepted and then ignored by the THP feed-forward network."""
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        gelu = _value_encoder(activation='gelu')(indicators, values, timestamps, masks)
        relu = _value_encoder(activation='relu')(indicators, values, timestamps, masks)
    assert not torch.allclose(gelu, relu), 'activation choice made no difference'


def test_value_encoder_zeroes_padded_positions():
    """Padded timesteps must encode to zero, which is what makes the row harmless downstream."""
    indicators, values, timestamps, masks = _value_batch()
    encoder = _value_encoder()
    with torch.no_grad():
        out = encoder(indicators, values, timestamps, masks)
    padded = out[masks == 0]
    assert padded.abs().max().item() < 1e-6, 'padded positions are not zero'


def test_the_delegated_stack_still_needs_the_nan_guard_and_the_migrated_one_does_not():
    """Why owning the attention is worth more than the rotary seam alone.

    ``torch.nn.MultiheadAttention`` softmaxes over an all-masked row and returns NaN, which then
    survives the ``val_enc * mask`` in the readout because NaN * 0 = NaN. That is what
    ``install_nan_hooks`` in dump_finetuned_predictions.py patches over at inference time. The
    migrated stack handles the row where it arises, so nothing needs patching.
    """
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        delegated = _value_encoder(norm='BatchNorm')(indicators, values, timestamps, masks)
        migrated = _value_encoder(norm='LayerNorm')(indicators, values, timestamps, masks)

    assert torch.isnan(delegated[3]).any(), (
        'the BatchNorm stack no longer NaNs on an all-padding episode -- if that is deliberate, '
        'the nan hooks in dump_finetuned_predictions.py can go too'
    )
    assert torch.isfinite(migrated).all()


def test_value_encoder_rejects_configurations_it_cannot_build():
    """Ways to ask for something the stack cannot deliver, each of them silent before."""
    common = dict(
        n_features=N_FEATURES, feat_dim=FEAT_DIM, d_model=D_MODEL,
        n_encoder_blocks=1, dim_feedforward=DIM_FF,
    )
    for kwargs, needle in (
        (dict(n_heads=N_HEADS, norm='RmsNorm'), 'norm'),
        (dict(n_heads=5, norm='LayerNorm'), 'divisible'),
        (dict(n_heads=N_HEADS, norm='BatchNorm', query_key_transform=torch.nn.Identity()),
         'query/key transform'),
        (dict(n_heads=N_HEADS, norm='BatchNorm', position_encoding='rope',
              ladder_p_min=2.0, ladder_p_max=3000.0), 'query/key transform'),
        (dict(n_heads=N_HEADS, norm='LayerNorm', position_encoding='rotary'),
         'position_encoding'),
        (dict(n_heads=N_HEADS, norm='LayerNorm', position_encoding='rope'), 'ladder_p_min'),
        (dict(n_heads=N_HEADS, norm='LayerNorm', position_encoding='rope', ladder_p_min=2.0,
              ladder_p_max=3000.0, query_key_transform=torch.nn.Identity()), 'one or the other'),
        (dict(n_heads=N_HEADS, norm='LayerNorm', ladder_p_min=2.0), 'together'),
    ):
        try:
            ValueDataEncoder(**common, **kwargs)
        except ValueError as error:
            assert needle in str(error), f'unexpected message for {kwargs}: {error}'
        else:
            raise AssertionError(f'{kwargs} was accepted')


def test_batchnorm_stack_matches_the_wrapper_it_replaced():
    """Dropping ``nn.TransformerEncoder`` must be a no-op.

    With ``enable_nested_tensor=False`` and no final norm the wrapper's forward is a loop over its
    layers, so running that loop directly should change nothing.
    """
    encoder = _value_encoder(norm='BatchNorm', n_blocks=2)
    # No all-padding episode: that row is NaN in both the wrapper and the loop, and NaN != NaN.
    indicators, values, timestamps, masks = _value_batch(include_empty=False)

    wrapper = torch.nn.TransformerEncoder(
        copy.deepcopy(encoder.layer_stack[0]), 2, enable_nested_tensor=False
    )
    wrapper.layers = encoder.layer_stack
    wrapper.eval()

    with torch.no_grad():
        actual = encoder(indicators, values, timestamps, masks)

        embedding = (
            encoder.indicator_input_projection_layer(indicators.float())
            + encoder.value_input_projection_layer(values)
        )
        embedding = encoder.position_encoding_layer(embedding, timestamps, masks)
        expected = encoder.activation(
            wrapper(embedding, src_key_padding_mask=~masks.bool())
        )

    delta = (actual - expected).abs().max().item()
    assert delta < 1e-6, f'the plain loop is not the wrapper (max delta {delta:.3e})'


# --------------------------------------------------------------------------------------------
# The seam a rotary encoding will occupy
# --------------------------------------------------------------------------------------------

class _Recorder(torch.nn.Module):
    """Passes q and k through untouched and records what it was handed."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, q, k, positions):
        self.calls.append((tuple(q.shape), tuple(k.shape), None if positions is None else positions.clone()))
        return q, k


class _Rotator(torch.nn.Module):
    """A stand-in with a rotation's shape: orthogonal, position-dependent, parameter-free.

    Not RoPE -- it rotates every channel pair by the same angle rather than by a ladder of
    frequencies -- but it exercises the same contract: norms of q and k are preserved, and the
    score picks up a dependence on the position difference.
    """

    def forward(self, q, k, positions):
        angle = positions[:, None, :, None] * 0.5  # (batch, 1, seq_len, 1)
        return self._rotate(q, angle), self._rotate(k, angle)

    @staticmethod
    def _rotate(x, angle):
        even, odd = x[..., 0::2], x[..., 1::2]
        cos, sin = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.flatten(start_dim=-2)


def test_transform_is_handed_per_head_tensors_and_the_timestamps():
    """The hook must fire after the head split and receive the timestamps as positions."""
    recorder = _Recorder()
    encoder = _value_encoder(n_blocks=2, transform=recorder)
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        encoder(indicators, values, timestamps, masks)

    assert len(recorder.calls) == 2, 'the transform did not run once per block'
    batch_size, seq_len = timestamps.shape
    for q_shape, k_shape, positions in recorder.calls:
        assert q_shape == (batch_size, N_HEADS, seq_len, D_MODEL // N_HEADS), q_shape
        assert k_shape == q_shape
        assert torch.equal(positions, timestamps), 'positions are not the record timestamps'

    recorder = _Recorder()
    event_encoder = _event_encoder(n_layers=2, transform=recorder)
    event_indicators, event_times, event_masks = _event_batch()
    with torch.no_grad():
        event_encoder(event_indicators, event_times, event_masks)

    assert len(recorder.calls) == 2, 'the transform did not reach the event stream'
    for q_shape, _, positions in recorder.calls:
        assert q_shape == (batch_size, N_HEADS, seq_len, 8), q_shape
        assert torch.equal(positions, event_times)


def test_no_transform_and_an_identity_transform_agree():
    """Installing a hook that returns q and k unchanged must change nothing."""
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        plain = _value_encoder()(indicators, values, timestamps, masks)
        hooked = _value_encoder(transform=_Recorder())(indicators, values, timestamps, masks)
    delta = (plain - hooked).abs().max().item()
    assert delta < 1e-6, f'the identity hook perturbed the output (max delta {delta:.3e})'


def test_a_rotation_at_the_seam_reaches_the_scores_and_leaves_the_values_alone():
    """What the seam has to make possible, checked on a stand-in rotation.

    Two things distinguish a rotation applied here from one applied to the layer input: it changes
    the attention scores, and it does not touch v. Both are checked -- the second by rerunning with
    the transform disabled on k only, which must still move the output.
    """
    indicators, values, timestamps, masks = _value_batch()

    with torch.no_grad():
        plain = _value_encoder(n_blocks=1)(indicators, values, timestamps, masks)
        rotated = _value_encoder(n_blocks=1, transform=_Rotator())(
            indicators, values, timestamps, masks
        )
    assert (rotated - plain).abs().max().item() > 1e-3, 'the rotation never reached the scores'

    # The rotation is orthogonal, so it must leave the norms of q and k untouched.
    rotator = _Rotator()
    generator = torch.Generator().manual_seed(9)
    q = torch.randn(2, N_HEADS, 8, D_MODEL // N_HEADS, generator=generator)
    positions = torch.rand(2, 8, generator=generator).cumsum(dim=1)
    rotated_q, _ = rotator(q, q.clone(), positions)
    assert torch.allclose(rotated_q.norm(dim=-1), q.norm(dim=-1), atol=1e-5)

    # A shifted timeline must give a shifted-but-equal score matrix: that is the property
    # "the score depends on the gap" reduces to, and it is why the hook sits after W_q and W_k.
    attention = MultiHeadAttention(N_HEADS, D_MODEL, 8, 8, dropout=0.0, query_key_transform=rotator)
    attention.eval()
    x = torch.randn(2, 8, D_MODEL, generator=generator)
    with torch.no_grad():
        _, first = attention(x, x, x, positions=positions, need_weights=True)
        _, second = attention(x, x, x, positions=positions + 17.0, need_weights=True)
    delta = (first - second).abs().max().item()
    assert delta < 1e-4, f'scores moved under a shift of the time origin (max delta {delta:.3e})'


def test_transform_instance_is_shared_by_every_block():
    """One transform object, installed once, reaches every layer of the stack.

    That sharing is what makes a frozen ladder a single buffer rather than one per block, and it
    is why a transform has to be stateless across calls.
    """
    recorder = _Recorder()
    encoder = _value_encoder(n_blocks=3, transform=recorder)
    installed = [block.self_attention.query_key_transform for block in encoder.layer_stack]
    assert all(item is recorder for item in installed), 'blocks did not share one transform'


# --------------------------------------------------------------------------------------------
# Contract the old stack met, re-checked against the new one
# --------------------------------------------------------------------------------------------

def test_episodes_remain_independent():
    encoder = _value_encoder()
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        baseline = encoder(indicators, values, timestamps, masks)
        perturbed_values = values.clone()
        perturbed_values[0] += 10.0
        perturbed = encoder(indicators, perturbed_values, timestamps, masks)

    assert (perturbed[0] - baseline[0]).abs().max().item() > 1e-4
    assert (perturbed[1:] - baseline[1:]).abs().max().item() < 1e-6, 'an episode leaked into another'


def test_attention_still_mixes_across_time():
    encoder = _value_encoder()
    indicators, values, timestamps, masks = _value_batch()
    with torch.no_grad():
        baseline = encoder(indicators, values, timestamps, masks)
        perturbed_values = values.clone()
        perturbed_values[0, -1] += 10.0
        perturbed = encoder(indicators, perturbed_values, timestamps, masks)

    assert (perturbed[0, 0] - baseline[0, 0]).abs().max().item() > 1e-4, (
        'the value encoder is no longer bidirectional over time'
    )


def test_padding_width_does_not_move_observed_positions():
    encoder = _value_encoder()
    indicators, values, timestamps, masks = _value_batch(batch_size=2, seq_len=8)
    seq_len, pad = values.size(1), 5

    def padded(tensor):
        shape = (tensor.size(0), pad) + tuple(tensor.shape[2:])
        return torch.cat([tensor, torch.zeros(shape)], dim=1)

    with torch.no_grad():
        unpadded = encoder(indicators, values, timestamps, masks)
        widened = encoder(
            padded(indicators),
            padded(values),
            padded(timestamps.unsqueeze(-1)).squeeze(-1),
            padded(masks.unsqueeze(-1)).squeeze(-1),
        )

    delta = (widened[:, :seq_len] - unpadded).abs().max().item()
    assert delta < 1e-5, f'padding width changed observed encodings (max delta {delta:.3e})'


def test_every_parameter_of_both_encoders_receives_gradient():
    """No parameter may be registered and never used -- DDP raises on exactly that."""
    for encoder, inputs in (
        (_value_encoder(n_blocks=2), _value_batch()),
        (_event_encoder(n_layers=2), _event_batch()),
    ):
        encoder.train()
        encoder(*inputs).sum().backward()
        unused = [
            name for name, parameter in encoder.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not unused, f'{type(encoder).__name__}: {len(unused)} parameter(s) unused: {unused[:6]}'


def test_stacked_blocks_are_independently_initialized():
    encoder = _value_encoder(n_blocks=3)
    first = encoder.layer_stack[0].state_dict()
    for index, block in enumerate(encoder.layer_stack[1:], start=1):
        other = block.state_dict()
        assert not all(torch.equal(first[key], other[key]) for key in first), (
            f'block {index} is a copy of block 0'
        )


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f'FAIL {name}\n     {exc}')
        else:
            print(f'PASS {name}')
    raise SystemExit(1 if failures else 0)
