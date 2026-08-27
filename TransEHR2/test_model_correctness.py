"""Probes for the model-correctness fixes.

These target failures that are silent at the loss curve: the model trains, the loss falls, and
the numbers are meaningless. Each probe is written to fail against the unfixed code, so running
it before the corresponding fix is part of using it.

Run directly (``python -m TransEHR2.test_model_correctness``) or under pytest.
"""

import torch

from TransEHR2.layers import TransformerBatchNormEncoderLayer
from TransEHR2.modules import ValueDataEncoder


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------

N_FEATURES = 6
FEAT_DIM = 6
D_MODEL = 16
N_HEADS = 2
N_BLOCKS = 1
DIM_FF = 16


def _build_value_encoder(
    seed: int = 0, n_blocks: int = N_BLOCKS, norm: str = 'LayerNorm'
) -> ValueDataEncoder:
    """A small ValueDataEncoder in eval mode, so dropout does not perturb comparisons."""
    torch.manual_seed(seed)
    encoder = ValueDataEncoder(
        n_features=N_FEATURES,
        feat_dim=FEAT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_blocks=n_blocks,
        dim_feedforward=DIM_FF,
        dropout=0.1,
        activation='gelu',
        norm=norm,
        normalize_before=True,
    )
    encoder.eval()
    return encoder


def _synthetic_batch(batch_size: int = 4, seq_len: int = 8, seed: int = 1):
    """Indicators, values, timestamps and an all-observed mask.

    Timestamps ascend by one hour per step, matching the hourly resample of the real data.
    """
    generator = torch.Generator().manual_seed(seed)
    indicators = (torch.rand(batch_size, seq_len, N_FEATURES, generator=generator) > 0.5).float()
    values = torch.randn(batch_size, seq_len, FEAT_DIM, generator=generator)
    timestamps = torch.arange(seq_len, dtype=torch.float32).expand(batch_size, seq_len).clone()
    masks = torch.ones(batch_size, seq_len)
    return indicators, values, timestamps, masks


# --------------------------------------------------------------------------------------------
# Fix 01 -- value encoder attention axis
# --------------------------------------------------------------------------------------------

def test_episodes_are_independent():
    """Perturbing one episode must not change any other episode's encoding.

    Fails against the unfixed encoder: the permute to (seq, batch, d) fed a layer built with
    ``batch_first=True``, so attention ran across the batch axis and every episode's encoding
    depended on the other episodes that happened to be batched with it.
    """
    encoder = _build_value_encoder()
    indicators, values, timestamps, masks = _synthetic_batch()

    with torch.no_grad():
        baseline = encoder(indicators, values, timestamps, masks)

        perturbed_values = values.clone()
        perturbed_values[0] += 10.0
        perturbed = encoder(indicators, perturbed_values, timestamps, masks)

    moved = (perturbed[0] - baseline[0]).abs().max().item()
    leaked = (perturbed[1:] - baseline[1:]).abs().max().item()

    assert moved > 1e-4, f'perturbing episode 0 did not change its own encoding (max delta {moved:.3e})'
    assert leaked < 1e-6, f'episode 0 leaked into other episodes (max delta {leaked:.3e})'


def test_attention_mixes_across_time():
    """A later timestep must influence earlier ones -- the encoder is bidirectional.

    Fails against the unfixed encoder, where the sequence axis was the batch axis: perturbing a
    later timestep changed only that timestep, because no attention ever ran over time.
    """
    encoder = _build_value_encoder()
    indicators, values, timestamps, masks = _synthetic_batch()
    late_step = values.size(1) - 1

    with torch.no_grad():
        baseline = encoder(indicators, values, timestamps, masks)

        perturbed_values = values.clone()
        perturbed_values[0, late_step] += 10.0
        perturbed = encoder(indicators, perturbed_values, timestamps, masks)

    earlier = (perturbed[0, 0] - baseline[0, 0]).abs().max().item()

    assert earlier > 1e-4, (
        f'perturbing timestep {late_step} did not reach timestep 0 (max delta {earlier:.3e}); '
        'attention is not running over the time axis'
    )


def test_padding_does_not_change_observed_positions():
    """Trailing padding must not alter the encoding of the observed timesteps.

    This is the mask-orientation half of the same bug: a padding mask shaped for the wrong axis
    masks the wrong thing, and the symptom is that padding width changes real outputs.
    """
    encoder = _build_value_encoder()
    indicators, values, timestamps, masks = _synthetic_batch(seq_len=8)
    batch_size, seq_len, _ = values.shape
    pad = 5

    def padded(tensor, fill=0.0):
        shape = (batch_size, pad) + tuple(tensor.shape[2:])
        return torch.cat([tensor, torch.full(shape, fill)], dim=1)

    with torch.no_grad():
        unpadded = encoder(indicators, values, timestamps, masks)
        padded_out = encoder(
            padded(indicators),
            padded(values),
            padded(timestamps.unsqueeze(-1)).squeeze(-1),
            padded(masks.unsqueeze(-1)).squeeze(-1),
        )

    delta = (padded_out[:, :seq_len] - unpadded).abs().max().item()
    assert delta < 1e-5, f'padding width changed observed-position encodings (max delta {delta:.3e})'


# --------------------------------------------------------------------------------------------
# Encoder stack construction
# --------------------------------------------------------------------------------------------

def test_encoder_blocks_are_independently_initialized():
    """Stacked blocks must not start life as copies of one another.

    ``nn.TransformerEncoder`` clones a single initialized prototype with ``copy.deepcopy``, so
    every block starts from identical weights. That is a torch quirk rather than a deliberate
    choice here, and it makes a deep stack behave differently from a freshly constructed one.
    """
    encoder = _build_value_encoder(n_blocks=3)
    layers = list(encoder.transformer_encoder.layers)
    assert len(layers) == 3

    first = layers[0].state_dict()
    for index, layer in enumerate(layers[1:], start=1):
        other = layer.state_dict()
        identical = all(torch.equal(first[key], other[key]) for key in first)
        assert not identical, (
            f'block {index} is a copy of block 0; stacked blocks share an initialization'
        )


def test_every_parameter_receives_gradient():
    """No parameter may be registered but never used.

    Unused parameters are wasted optimizer state and checkpoint weight, and under DDP they raise
    at the reduction step unless find_unused_parameters is set. This is the general guard; it
    caught the prototype layer that nn.TransformerEncoder leaves behind after cloning.
    """
    encoder = _build_value_encoder(n_blocks=2)
    encoder.train()
    indicators, values, timestamps, masks = _synthetic_batch()

    encoder(indicators, values, timestamps, masks).sum().backward()

    unused = [name for name, p in encoder.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f'{len(unused)} parameter(s) never received a gradient: {unused[:6]}'


# --------------------------------------------------------------------------------------------
# Fix 02 -- generator mask polarity
# --------------------------------------------------------------------------------------------

class _CapturingEncoder(torch.nn.Module):
    """Stands in for the value encoder and records exactly what the generator handed it."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.seen = {}

    def forward(self, indicators, values, timestamps, timestep_masks):
        self.seen['indicators'] = indicators.detach().clone()
        self.seen['values'] = values.detach().clone()
        return torch.zeros(values.size(0), values.size(1), self.d_model)


def test_generator_input_is_masked_not_revealed():
    """The generator must see zeros where components are masked, and the data everywhere else.

    ``generate_record_masks`` marks masked components with ones. Multiplying the data by that
    mask therefore kept precisely the components the generator is asked to reconstruct and
    zeroed all of its context -- the reconstruction target was fed in as input. The loss still
    fell, which is why this never surfaced.
    """
    from TransEHR2.modules import MaskedTokenGenerator

    torch.manual_seed(0)
    batch_size, seq_len, dims = 3, 5, [2, 3]
    capture = _CapturingEncoder()
    generator = MaskedTokenGenerator(
        encoder=capture,
        d_model=D_MODEL,
        numeric_dims=dims,
        categorical_classes=[],
    )
    generator.eval()

    generator_ = torch.Generator().manual_seed(7)
    values = [torch.randn(batch_size, seq_len, d, generator=generator_) + 5.0 for d in dims]
    indicators = torch.ones(batch_size, seq_len, len(dims))
    value_masks = [
        (torch.rand(batch_size, seq_len, d, generator=generator_) > 0.5).float() for d in dims
    ]
    indicator_masks = (torch.rand(batch_size, seq_len, len(dims), generator=generator_) > 0.5).float()

    batch = {
        'numeric': {'indicators': indicators, 'values': values},
        'times': torch.arange(seq_len, dtype=torch.float32).expand(batch_size, seq_len).clone(),
        'masks': torch.ones(batch_size, seq_len),
    }
    record_masks = {'numeric': {'indicators': indicator_masks, 'values': value_masks}}

    with torch.no_grad():
        generator(batch, record_masks)

    original = torch.cat(values, dim=2)
    mask = torch.cat(value_masks, dim=2)
    seen = capture.seen['values']

    revealed = seen[mask.bool()].abs().max().item()
    assert revealed == 0.0, (
        f'generator was shown {int(mask.sum().item())} masked components it is meant to predict '
        f'(max magnitude {revealed:.3f})'
    )

    context_delta = (seen[~mask.bool()] - original[~mask.bool()]).abs().max().item()
    assert context_delta < 1e-6, (
        f'generator context was altered where it should be intact (max delta {context_delta:.3e})'
    )


# --------------------------------------------------------------------------------------------
# BatchNorm encoder path
# --------------------------------------------------------------------------------------------

def test_batchnorm_encoder_builds_and_runs():
    """``norm='BatchNorm'`` must construct and complete a forward pass.

    Fails against the unfixed builder, which passed ``batch_first=True`` to both branches;
    TransformerBatchNormEncoderLayer takes no such argument, so construction raised TypeError.
    The path is inert in the shipped configs -- every experiment sets NORM: 'LayerNorm' -- so
    nothing exercised it.
    """
    encoder = _build_value_encoder(norm='BatchNorm')
    layers = list(encoder.transformer_encoder.layers)
    assert all(isinstance(layer, TransformerBatchNormEncoderLayer) for layer in layers)

    indicators, values, timestamps, masks = _synthetic_batch()
    with torch.no_grad():
        output = encoder(indicators, values, timestamps, masks)

    assert output.shape == (values.size(0), values.size(1), D_MODEL)
    assert torch.isfinite(output).all(), 'BatchNorm encoder produced non-finite activations'


def test_batchnorm_normalizes_over_batch_and_time_per_feature():
    """The layer's permutes must put d_model on the channel axis for a batch-first input.

    The permutes and docstrings in TransformerBatchNormEncoderLayer were written for
    (seq, batch, d_model) inputs, but the encoder now feeds (batch, seq, d_model). This pins
    down that the running statistics are still one per feature, pooled over every
    (episode, timestep) position, and that the layer returns the layout it was given.
    """
    torch.manual_seed(0)
    layer = TransformerBatchNormEncoderLayer(
        D_MODEL, N_HEADS, DIM_FF, dropout=0.0, activation='gelu', norm_first=True
    )
    layer.train()

    batch_size, seq_len = 4, 8
    src = torch.randn(batch_size, seq_len, D_MODEL, generator=torch.Generator().manual_seed(2))
    output = layer(src)

    assert output.shape == src.shape, 'layer did not return its input layout'
    assert layer.norm1.running_mean.numel() == D_MODEL, 'channel axis is not d_model'

    # norm_first applies norm1 to src unchanged, so with momentum 0.1 and zeroed running
    # statistics the update is 0.1 * the batch statistic over all (episode, timestep) positions.
    pooled = src.reshape(-1, D_MODEL)
    assert torch.allclose(layer.norm1.running_mean, 0.1 * pooled.mean(0), atol=1e-6), (
        'running mean is not the per-feature mean pooled over episodes and timesteps'
    )
    assert torch.allclose(
        layer.norm1.running_var, 0.9 + 0.1 * pooled.var(0, unbiased=True), atol=1e-6
    ), 'running variance is not the per-feature variance pooled over episodes and timesteps'


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


# --------------------------------------------------------------------------------------------
# Fix 03 -- THP restricted to in-stay records
# --------------------------------------------------------------------------------------------

HIST_LEN = 6
EPISODE_LEN = 4
N_EVENT_FEATS = 3


def _episode(n_hist_records: int):
    """One episode dict in the stored extraction layout.

    History is right-justified in ``[0, HIST_LEN)`` and in-stay data is left-justified from
    ``HIST_LEN``, so an episode holding fewer than ``HIST_LEN`` historical records carries a run
    of leading padding -- the case the THP indexing bug turns on.
    """
    ts = HIST_LEN + EPISODE_LEN
    masks = torch.zeros(ts)
    times = torch.zeros(ts)

    if n_hist_records:
        lo = HIST_LEN - n_hist_records
        masks[lo:HIST_LEN] = 1.0
        # Historical records sit far in the past, on no particular grid.
        times[lo:HIST_LEN] = torch.linspace(-5000.0, -100.0, n_hist_records)

    # In-stay records are left-justified from HIST_LEN and lie on the 1 h resample grid.
    masks[HIST_LEN:] = 1.0
    times[HIST_LEN:] = torch.arange(float(EPISODE_LEN))

    event_ind = torch.zeros(ts, N_EVENT_FEATS)
    event_ind[masks.bool()] = 1.0

    return {
        'val_numeric_indicators': torch.zeros(ts, 1),
        'val_numeric_values': [torch.zeros(ts, 1)],
        'val_categorical_indicators': torch.zeros(ts, 0),
        'val_categorical_values': [],
        'val_ordinal_indicators': torch.empty(0),
        'val_ordinal_values': [],
        'val_multilabel_indicators': torch.empty(0),
        'val_multilabel_values': [],
        'val_text_indicators': torch.zeros(ts, 0),
        'val_text_embeddings': [],
        'val_times': times.clone(),
        'val_masks': masks.clone(),
        'event_indicators': event_ind,
        'event_times': times.clone(),
        'event_masks': masks.clone(),
        'static_data': torch.zeros(2),
        'mortality': torch.tensor(0.0),
        'length_of_stay': torch.tensor(1.0),
        'phenotype': torch.zeros(3),
    }


def _collated():
    from TransEHR2.data.preprocessing import collate_tensorized

    # A full history window, a partial one, and none at all -- the last two carry leading padding.
    batch = [_episode(HIST_LEN), _episode(2), _episode(0)]
    return collate_tensorized(batch, max_history_len_steps=HIST_LEN)


def test_event_stream_drops_the_history_region():
    """The event stream the THP sees must be in-stay records only."""

    out = _collated()
    for key in ('indicators', 'times', 'masks'):
        assert out['event_data'][key].shape[1] == EPISODE_LEN, (
            f"event_data['{key}'] is {out['event_data'][key].shape[1]} wide, "
            f'expected {EPISODE_LEN}'
        )
    # History still reaches the value encoder.
    assert out['val_data']['masks'].shape[1] == HIST_LEN + EPISODE_LEN


def test_first_event_index_is_observed_for_every_episode():
    """The THP gates its base intensity on index 0, so index 0 must never be padding.

    This is what fails on the unfixed code: an episode with fewer than HIST_LEN historical
    records has padding at index 0, so `initial_non_event_ll` is multiplied by zero and the
    base-intensity term silently vanishes.
    """

    first = _collated()['event_data']['masks'][:, 0]
    assert torch.all(first == 1.0), f'padding at event index 0 for some episode: {first}'


def test_in_stay_gaps_are_on_the_resample_grid():
    """Restricting to in-stay records removes the history/in-stay boundary delta."""

    out = _collated()
    times, masks = out['event_data']['times'], out['event_data']['masks']
    both_valid = masks[:, 1:] * masks[:, :-1]
    gaps = (times[:, 1:] - times[:, :-1]) * both_valid
    assert gaps.max().item() <= 1.0, f'inter-event gap of {gaps.max().item()} h survived'


# --------------------------------------------------------------------------------------------
# Fix 04 -- max readout is padding-invariant
# --------------------------------------------------------------------------------------------

class _FixedEncoder(torch.nn.Module):
    """Returns an encoding determined by the timestamp, so padding cannot change observed rows."""

    def forward(self, indicators, times, masks):
        # Every observed position gets a strictly negative encoding: -(1 + t).
        return -(1.0 + times)[..., None].expand(-1, -1, D_MODEL).contiguous()


def _max_readout(masks, times):
    """Run MixedClassifier with aggr='max' and capture what reaches the output head."""
    from TransEHR2.models import MixedClassifier

    model = MixedClassifier(
        event_encoder=_FixedEncoder(),
        val_encoder=_FixedEncoder(),
        d_event_enc=D_MODEL,
        d_val_enc=0,
        d_statics=0,
        num_classes=2,
        aggr='max',
    )
    captured = {}

    def _capture(module, inputs, output):
        # Must return None: a forward hook's return value replaces the module's output.
        captured['x'] = inputs[0].detach()

    model.linear.register_forward_hook(_capture)
    batch = {
        'event_data': {
            'indicators': torch.zeros(masks.shape[0], masks.shape[1], N_EVENT_FEATS),
            'times': times,
            'masks': masks,
        }
    }
    model(batch)
    return captured['x']


def test_max_readout_never_returns_a_value_no_record_produced():
    """With every observed encoding negative, a zeroed-padding max returns 0 -- from nothing."""

    masks = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    times = torch.tensor([[1.0, 2.0, 3.0, 0.0]])

    readout = _max_readout(masks, times)
    # Observed encodings are -2, -3, -4; the max over observed records is -2.
    assert torch.allclose(readout, torch.full_like(readout, -2.0)), (
        f'max readout returned {readout.flatten()[0].item()}, expected -2.0 '
        '(0.0 means padding won the max)'
    )


def test_max_readout_is_invariant_to_padding_width():
    """The same observed records must give the same readout however much padding follows."""

    short = _max_readout(
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 3.0, 0.0]]),
    )
    long = _max_readout(
        torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(short, long)


def test_max_readout_gives_all_padding_rows_the_zero_vector():
    """An episode with no observed records must not come back as -inf."""

    readout = _max_readout(torch.zeros(1, 4), torch.zeros(1, 4))
    assert torch.all(torch.isfinite(readout)), 'all-padding row leaked -inf'
    assert torch.allclose(readout, torch.zeros_like(readout))
