"""Probes for the model-correctness fixes.

These target failures that are silent at the loss curve: the model trains, the loss falls, and
the numbers are meaningless. Each probe is written to fail against the unfixed code, so running
it before the corresponding fix is part of using it.

Run directly (``python -m TransEHR2.test_model_correctness``) or under pytest.
"""

import torch

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


def _build_value_encoder(seed: int = 0, n_blocks: int = N_BLOCKS) -> ValueDataEncoder:
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
        norm='LayerNorm',
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
