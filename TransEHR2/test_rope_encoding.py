"""Probes for the re-spanned ladder and continuous-position RoPE.

Three claims are under test, and they are the ones the encoding arm rests on.

**The ladder spans the gaps the data actually contains.** The inherited `1e4` constant is a lambda
ladder running 1 to 9,330 -- band periods 6.28 h to 58,470 h -- chosen for token indices and never
checked against a quantity in hours. At the largest observed gap not one of its 128 bands is still
phase-coherent, so records months apart encode to mutually near-orthogonal vectors carrying no
recoverable ordering. `build_frequency_ladder` is checked against its formula, and against the
inherited ladder on exactly that failure.

**The rotation reaches the score as a function of the gap alone.** A rotation applied to q and k
after W_q and W_k makes `q_i . k_j` depend on `t_i - t_j` and nothing else, which is testable
directly: shift the whole timeline and the attention must not move. The additive arm must move,
because its encoding is a function of absolute time -- that is the difference the arm comparison is
about, and both halves are asserted.

**The two arms differ in their encoding and in nothing else.** Same ladder bounds, same band count
where the projection is square, both parameter-free, dropout and the padding mask retained on both
sides. An arm comparison that also varied regularization or band count would not be measuring the
mechanism.

Run directly (``python -m TransEHR2.test_rope_encoding``) or under pytest.
"""

import math

import torch

from TransEHR2.layers import (
    MultiHeadAttention,
    RotaryTemporalEncoding,
    TemporalPositionEncoding,
    build_frequency_ladder,
)
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder


# The production geometry, from experiment1_baseline.yaml.
D_MODEL = 256
N_HEADS = 2
THP_D_K = 128

# Ladder bounds from the plan: P_min = 2 * delta_min, P_max = 63 * delta_max.
VALUE_GAP_MAX = 125_073.0
VALUE_P_MIN, VALUE_P_MAX = 2.0, 7.9e6
EVENT_P_MIN, EVENT_P_MAX = 2.0, 3000.0

# Small shapes for the end-to-end probes.
N_FEATURES = 6
FEAT_DIM = 6
N_EVENT_TYPES = 5
SMALL_D_MODEL = 16
DIM_FF = 16


def _value_encoder(scheme='rope', p_min=EVENT_P_MIN, p_max=EVENT_P_MAX, seed=0, n_blocks=2):
    torch.manual_seed(seed)
    encoder = ValueDataEncoder(
        n_features=N_FEATURES, feat_dim=FEAT_DIM, d_model=SMALL_D_MODEL, n_heads=N_HEADS,
        n_encoder_blocks=n_blocks, dim_feedforward=DIM_FF, dropout=0.1, activation='gelu',
        norm='LayerNorm', normalize_before=True, position_encoding=scheme,
        ladder_p_min=p_min, ladder_p_max=p_max,
    )
    encoder.eval()
    return encoder


def _event_encoder(scheme='rope', p_min=EVENT_P_MIN, p_max=EVENT_P_MAX, seed=0, n_layers=2):
    torch.manual_seed(seed)
    encoder = EventDataEncoder(
        num_types=N_EVENT_TYPES, d_model=SMALL_D_MODEL, d_inner=DIM_FF, n_layers=n_layers,
        n_head=N_HEADS, d_k=8, d_v=8, dropout=0.1, normalize_before=True,
        position_encoding=scheme, ladder_p_min=p_min, ladder_p_max=p_max,
    )
    encoder.eval()
    return encoder


def _batch(batch_size=4, seq_len=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    indicators = (torch.rand(batch_size, seq_len, N_FEATURES, generator=generator) > 0.5).float()
    values = torch.randn(batch_size, seq_len, FEAT_DIM, generator=generator)
    timestamps = (torch.rand(batch_size, seq_len, generator=generator) * 3.0).cumsum(dim=1)
    masks = torch.ones(batch_size, seq_len)
    masks[1, -3:] = 0.0
    masks[2, :2] = 0.0
    return indicators, values, timestamps, masks


def _event_inputs(batch_size=4, seq_len=8, seed=2):
    generator = torch.Generator().manual_seed(seed)
    indicators = (torch.rand(batch_size, seq_len, N_EVENT_TYPES, generator=generator) > 0.5).float()
    timestamps = (torch.rand(batch_size, seq_len, generator=generator) * 3.0).cumsum(dim=1)
    masks = torch.ones(batch_size, seq_len)
    masks[1, -3:] = 0.0
    return indicators, timestamps, masks


def _informative_bands(lambdas, gap):
    """How many bands say something usable about a gap: 0.1 <= gap / lambda <= pi."""
    phase = gap / lambdas
    return int(((phase >= 0.1) & (phase <= math.pi)).sum())


# --------------------------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------------------------

def test_ladder_is_log_uniform_between_its_endpoints():
    lambdas = build_frequency_ladder(128, VALUE_P_MIN, VALUE_P_MAX)
    periods = lambdas * 2.0 * math.pi

    assert lambdas.shape == (128,)
    assert abs(periods[0].item() - VALUE_P_MIN) < 1e-3, 'first band is not P_min'
    assert abs(periods[-1].item() / VALUE_P_MAX - 1.0) < 1e-6, 'last band is not P_max'
    # Log-uniform means a constant ratio between neighbours.
    ratios = (periods[1:] / periods[:-1]).log()
    assert (ratios - ratios.mean()).abs().max().item() < 1e-4, 'spacing is not log-uniform'


def test_ladder_stays_coherent_across_the_observed_gap_range_and_the_inherited_one_does_not():
    """The defect the re-spanning exists to fix, stated as a test.

    A band is wrapped once the phase change over a gap exceeds pi. Counting bands that are neither
    wrapped nor frozen at the largest observed gap is the whole diagnosis: the inherited ladder has
    none, so history months apart is encoded with no recoverable ordering.
    """
    respanned = build_frequency_ladder(D_MODEL // 2, VALUE_P_MIN, VALUE_P_MAX)
    inherited = torch.exp(
        torch.arange(0, D_MODEL, 2, dtype=torch.float32) * math.log(1e4) / D_MODEL
    )

    assert _informative_bands(inherited, VALUE_GAP_MAX) == 0, (
        'the inherited ladder is no longer the broken baseline this test assumes'
    )
    assert _informative_bands(respanned, VALUE_GAP_MAX) > 0, (
        f'no band survives the largest observed gap ({VALUE_GAP_MAX} h)'
    )
    # And at the other end of the range, where the inherited ladder was fine, so is this one.
    assert _informative_bands(respanned, 1.0) > 0


def test_informative_band_counts_match_the_plan():
    """Each band covers ~1.50 decades of gap, so the count per scale follows from the span.

    Recorded rather than derived: these are the numbers the ladder table in the revision plan
    quotes, and a change to the bounds should show up here rather than in a training run.
    """
    value = build_frequency_ladder(D_MODEL // 2, VALUE_P_MIN, VALUE_P_MAX)
    event = build_frequency_ladder(D_MODEL // 2, EVENT_P_MIN, EVENT_P_MAX)

    value_count = _informative_bands(value, 100.0)
    event_count = _informative_bands(event, 10.0)
    print(f'  informative bands per scale: value {value_count}, event {event_count}')

    assert 27 <= value_count <= 31, f'value ladder gives {value_count} bands per scale, expected ~29'
    assert 57 <= event_count <= 62, f'event ladder gives {event_count} bands per scale, expected ~59'
    assert event_count > value_count, (
        'the narrower event range must buy resolution -- that is the whole point of giving the '
        'event stream its own ladder post fix 03'
    )


def test_ladder_rejects_impossible_bounds():
    for kwargs, needle in (
        (dict(n_bands=0, p_min=2.0, p_max=10.0), 'n_bands'),
        (dict(n_bands=8, p_min=0.0, p_max=10.0), 'p_min'),
        (dict(n_bands=8, p_min=10.0, p_max=10.0), 'p_max'),
    ):
        try:
            build_frequency_ladder(**kwargs)
        except ValueError as error:
            assert needle in str(error), f'unexpected message for {kwargs}: {error}'
        else:
            raise AssertionError(f'{kwargs} was accepted')


# --------------------------------------------------------------------------------------------
# The rotation
# --------------------------------------------------------------------------------------------

def test_rotation_preserves_the_norms_of_q_and_k():
    """It is a rotation, so it spends no capacity: content survives it intact."""
    rotary = RotaryTemporalEncoding(N_HEADS, 128, VALUE_P_MIN, VALUE_P_MAX)
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(2, N_HEADS, 6, 128, generator=generator)
    k = torch.randn(2, N_HEADS, 6, 128, generator=generator)
    positions = torch.rand(2, 6, generator=generator).cumsum(dim=1) * 500.0

    rotated_q, rotated_k = rotary(q, k, positions)
    assert torch.allclose(rotated_q.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(rotated_k.norm(dim=-1), k.norm(dim=-1), atol=1e-4)
    # Pairwise, not just in aggregate: each channel pair is turned, none is rescaled.
    pair_norm = lambda x: (x[..., 0::2] ** 2 + x[..., 1::2] ** 2).sqrt()
    assert torch.allclose(pair_norm(rotated_q), pair_norm(q), atol=1e-4)


def test_the_score_depends_on_the_gap_and_not_on_the_time_origin():
    """The property the whole arm is built to have.

    Shifting every timestamp by a constant leaves every pairwise gap unchanged, so an encoding that
    reaches the score purely through the gap must leave the score matrix untouched. The shift is
    applied in float64 so that what is being tested is the encoding rather than the arithmetic of
    the shift itself -- see the next probe for what float32 storage costs.
    """
    rotary = RotaryTemporalEncoding(N_HEADS, 128, VALUE_P_MIN, VALUE_P_MAX)
    attention = MultiHeadAttention(N_HEADS, D_MODEL, 128, 128, dropout=0.0, query_key_transform=rotary)
    attention.eval()

    generator = torch.Generator().manual_seed(4)
    x = torch.randn(2, 10, D_MODEL, generator=generator)
    positions = (torch.rand(2, 10, generator=generator) * 4.0).cumsum(dim=1).double()

    with torch.no_grad():
        base = attention(x, x, x, positions=positions, need_weights=True)[1]
        for shift in (17.0, 5_000.0, VALUE_GAP_MAX):
            shifted = attention(x, x, x, positions=positions + shift, need_weights=True)[1]
            delta = (base - shifted).abs().max().item()
            assert delta < 1e-5, f'the score moved by {delta:.3e} under a shift of {shift} h'


def test_what_float32_timestamps_cost_the_invariance():
    """The residual drift is the timestamp's own resolution, not the rotation's.

    A float32 timestamp near 1.25e5 h resolves to 0.0078 h, so shifting the timeline that far and
    storing the result perturbs every gap by up to 28 seconds. Against the fastest band of the value
    ladder that is 0.025 rad of phase jitter. Two things follow, and both are worth having written
    down: the drift is bounded and small, and it is *arm-neutral* -- the additive encoding reads the
    same quantized timestamps -- so it cannot bias the comparison. It is, however, an argument for
    re-basing the time origin at admission if the fastest bands ever start to matter.
    """
    rotary = RotaryTemporalEncoding(N_HEADS, 128, VALUE_P_MIN, VALUE_P_MAX)
    attention = MultiHeadAttention(N_HEADS, D_MODEL, 128, 128, dropout=0.0, query_key_transform=rotary)
    attention.eval()

    generator = torch.Generator().manual_seed(4)
    x = torch.randn(2, 10, D_MODEL, generator=generator)
    positions = (torch.rand(2, 10, generator=generator) * 4.0).cumsum(dim=1)

    with torch.no_grad():
        base = attention(x, x, x, positions=positions, need_weights=True)[1]
        drift = {
            shift: (
                base - attention(x, x, x, positions=positions + shift, need_weights=True)[1]
            ).abs().max().item()
            for shift in (17.0, 5_000.0, VALUE_GAP_MAX)
        }
        exact = (
            base.double()
            - attention(
                x, x, x, positions=positions.double() + VALUE_GAP_MAX, need_weights=True
            )[1].double()
        ).abs().max().item()

    print('  attention-weight drift under a stored float32 shift: '
          + ', '.join(f'{shift:.0f} h -> {value:.1e}' for shift, value in drift.items())
          + f' (float64 shift at {VALUE_GAP_MAX:.0f} h -> {exact:.1e})')

    assert drift[VALUE_GAP_MAX] < 5e-3, (
        f'drift at the extreme is {drift[VALUE_GAP_MAX]:.3e}, larger than float32 storage explains'
    )
    assert exact < 1e-5, 'the drift is not float32 storage after all -- the rotation is losing phase'


def test_records_sharing_a_timestamp_are_rotated_identically():
    """Two records at the same instant must attend to each other as if unrotated.

    The hourly resample grid puts many records at the same timestamp, so this is the common case
    rather than a corner one.
    """
    rotary = RotaryTemporalEncoding(N_HEADS, 128, VALUE_P_MIN, VALUE_P_MAX)
    generator = torch.Generator().manual_seed(5)
    q = torch.randn(1, N_HEADS, 4, 128, generator=generator)
    k = torch.randn(1, N_HEADS, 4, 128, generator=generator)
    positions = torch.full((1, 4), 240.0)

    rotated_q, rotated_k = rotary(q, k, positions)
    assert torch.allclose(rotated_q @ rotated_k.transpose(-2, -1), q @ k.transpose(-2, -1), atol=1e-3)


def test_bands_are_partitioned_across_heads_not_repeated():
    """Head h takes the contiguous slice [h*B, (h+1)*B) of one ladder.

    Repeating the ladder per head would leave the rotary arm with half or a quarter of the additive
    arm's distinct frequencies, and a win for the additive arm would then be unattributable.
    """
    rotary = RotaryTemporalEncoding(N_HEADS, 128, VALUE_P_MIN, VALUE_P_MAX)
    full = build_frequency_ladder(N_HEADS * 64, VALUE_P_MIN, VALUE_P_MAX)

    assert rotary.lambdas.shape == (N_HEADS, 64)
    for head in range(N_HEADS):
        assert torch.equal(rotary.lambdas[head], full[head * 64:(head + 1) * 64])
    assert rotary.lambdas[0].max() < rotary.lambdas[1].min(), 'head slices overlap in frequency'
    assert torch.unique(rotary.lambdas).numel() == rotary.band_count, 'bands are repeated'


def test_band_count_matches_the_additive_arm_on_both_streams():
    """Parity needs `n_head * d_head == d_model`, and only one stream gets that for free.

    The value encoder derives its per-head width from `d_model`, so it cannot be wrong. The event
    encoder takes `d_k` from config, which is why `THP_ENCODER_D_K` is 128 rather than the
    submitted model's 64 -- at 64 the rotary arm ran 64 bands against the additive arm's 128, and
    a win for the additive arm could have been a win for twice the frequencies instead.
    """
    for label, d_head, bounds in (
        ('value', D_MODEL // N_HEADS, (VALUE_P_MIN, VALUE_P_MAX)),
        ('event', THP_D_K, (EVENT_P_MIN, EVENT_P_MAX)),
    ):
        rotary = RotaryTemporalEncoding(N_HEADS, d_head, *bounds)
        assert rotary.band_count == D_MODEL // 2, (
            f'{label} stream: rotary arm has {rotary.band_count} bands against the additive '
            f"arm's {D_MODEL // 2}"
        )


def test_every_shipped_config_gives_the_two_arms_the_same_band_count():
    """The parity check that reads the configs, so lowering d_k cannot pass unnoticed."""
    import pathlib

    import yaml

    directory = pathlib.Path(__file__).resolve().parent / 'configs' / 'experiments'
    checked = 0
    for path in sorted(directory.glob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        if 'THP_ENCODER_D_K' not in config:
            continue
        checked += 1
        additive_bands = config['THP_ENCODER_D_MODEL'] // 2
        rotary_bands = config['THP_ENCODER_N_HEADS'] * config['THP_ENCODER_D_K'] // 2
        assert rotary_bands == additive_bands, (
            f'{path.name}: event stream would run {rotary_bands} rotary bands against '
            f'{additive_bands} additive ones; THP_ENCODER_D_K must be '
            f"{config['THP_ENCODER_D_MODEL'] // config['THP_ENCODER_N_HEADS']}"
        )
    assert checked >= 8, f'only {checked} configs carry an event encoder; the scan is not working'
    print(f'  {checked} shipped configs, band parity on all of them')


def test_rotation_rejects_what_it_cannot_encode():
    try:
        RotaryTemporalEncoding(2, 65, 2.0, 10.0)
    except ValueError as error:
        assert 'even' in str(error)
    else:
        raise AssertionError('an odd d_head was accepted')

    rotary = RotaryTemporalEncoding(2, 8, 2.0, 10.0)
    q = torch.zeros(1, 2, 4, 8)
    for positions, needle in ((None, 'timestamps'), (torch.zeros(1, 5), 'sequence length')):
        try:
            rotary(q, q, positions)
        except ValueError as error:
            assert needle in str(error), f'unexpected message: {error}'
        else:
            raise AssertionError(f'positions={positions} was accepted')


# --------------------------------------------------------------------------------------------
# The arms, end to end
# --------------------------------------------------------------------------------------------

def test_the_rotary_arm_is_invariant_to_the_time_origin_and_the_additive_arm_is_not():
    """The same contrast as the score-level probe, now through a whole encoder.

    This is the factor the arm comparison isolates. Both arms see the same gaps; only the additive
    arm also sees where those gaps sit on the absolute clock.
    """
    indicators, values, timestamps, masks = _batch()
    shift = 10_000.0
    # In float64, so the shift itself is exact and the encoding is what is under test. What float32
    # storage adds on top is measured in test_what_float32_timestamps_cost_the_invariance.
    timestamps = timestamps.double()

    with torch.no_grad():
        rotary = _value_encoder(scheme='rope')
        base = rotary(indicators, values, timestamps, masks)
        shifted = rotary(indicators, values, timestamps + shift, masks)
        delta = (base - shifted).abs().max().item()
        assert delta < 1e-4, f'the rotary arm moved by {delta:.3e} under a shift of the time origin'

        additive = _value_encoder(scheme='additive')
        moved = (
            additive(indicators, values, timestamps, masks)
            - additive(indicators, values, timestamps + shift, masks)
        ).abs().max().item()
        assert moved > 1e-3, (
            'the additive arm did not move under a shift of the time origin -- if that is now true, '
            'the two arms no longer differ in the way the comparison claims'
        )
    print(f'  time-origin shift of {shift:.0f} h: rotary moved {delta:.2e}, additive moved {moved:.2e}')


def test_the_rotary_arm_reaches_the_event_stream_too():
    indicators, timestamps, masks = _event_inputs()
    encoder = _event_encoder(scheme='rope')
    timestamps = timestamps.double()
    with torch.no_grad():
        base = encoder(indicators, timestamps, masks)
        shifted = encoder(indicators, timestamps + 5_000.0, masks)
    assert torch.isfinite(base).all()
    assert (base - shifted).abs().max().item() < 1e-4, 'the event stream is not gap-only'


def test_both_arms_are_parameter_free_in_their_encoding():
    """What makes one hyperparameter grid defensible across both arms."""
    additive = _value_encoder(scheme='additive')
    rotary = _value_encoder(scheme='rope')

    assert sum(p.numel() for p in additive.parameters()) == sum(p.numel() for p in rotary.parameters())
    assert list(rotary.query_key_transform.parameters()) == []
    assert list(additive.position_encoding_layer.parameters()) == []


def test_the_rotary_arm_keeps_the_dropout_and_the_padding_mask():
    """Dropping the whole additive layer would make the comparison about regularization.

    `TemporalPositionEncoding` does add -> dropout -> mask as one unit. The rotary arm keeps the
    last two: the dropout so both arms carry the same regularizer, and the mask so a padded record
    -- whose stored timestamp is 0.0, a legitimate admission time -- carries a zero vector into q
    and k rather than a rotation by zero.
    """
    encoder = _value_encoder(scheme='rope')
    layer = encoder.position_encoding_layer
    assert layer.additive is False
    assert isinstance(layer.dropout, torch.nn.Dropout) and layer.dropout.p == 0.1

    x = torch.ones(2, 5, SMALL_D_MODEL)
    masks = torch.ones(2, 5)
    masks[0, 3:] = 0.0
    times = torch.arange(5, dtype=torch.float32).expand(2, 5).clone()
    layer.eval()
    with torch.no_grad():
        out = layer(x, times, masks)
    assert torch.equal(out[masks == 0], torch.zeros_like(out[masks == 0])), 'padding is not zeroed'
    assert torch.equal(out[masks == 1], x[masks == 1]), 'the additive term was not switched off'


def test_neither_ladder_is_written_into_a_checkpoint():
    """Re-spanning changes the values but not the shape, so a persistent buffer would be silent.

    A checkpoint from the inherited ladder loads into a re-spanned encoder without complaint and
    without the shape mismatch that would otherwise flag it -- and the arm running would not be the
    arm intended.
    """
    respanned = _value_encoder(scheme='additive')
    legacy = ValueDataEncoder(
        n_features=N_FEATURES, feat_dim=FEAT_DIM, d_model=SMALL_D_MODEL, n_heads=N_HEADS,
        n_encoder_blocks=2, dim_feedforward=DIM_FF, dropout=0.1, activation='gelu',
        norm='LayerNorm', normalize_before=True,
    )

    assert 'position_encoding_layer.position_encoding' not in respanned.state_dict()
    assert not any(key.endswith('lambdas') for key in _value_encoder(scheme='rope').state_dict())

    ladder_before = respanned.position_encoding_layer.position_encoding.clone()
    result = respanned.load_state_dict(legacy.state_dict(), strict=True)
    assert torch.equal(respanned.position_encoding_layer.position_encoding, ladder_before), (
        'a checkpoint overwrote the re-spanned ladder'
    )
    assert not result.unexpected_keys, result.unexpected_keys


def test_an_unbounded_additive_encoder_is_the_inherited_ladder_untouched():
    """Omitting the bounds must change nothing, so an existing config is unaffected."""
    encoder = ValueDataEncoder(
        n_features=N_FEATURES, feat_dim=FEAT_DIM, d_model=SMALL_D_MODEL, n_heads=N_HEADS,
        n_encoder_blocks=1, dim_feedforward=DIM_FF, norm='LayerNorm', normalize_before=True,
    )
    inherited = torch.exp(
        torch.arange(0, SMALL_D_MODEL, 2, dtype=torch.float32) * math.log(1e4) / SMALL_D_MODEL
    )
    assert torch.equal(encoder.position_encoding_layer.position_encoding, inherited)
    assert encoder.query_key_transform is None


def test_both_arms_train_and_reach_every_parameter():
    for scheme in ('additive', 'rope'):
        encoder = _value_encoder(scheme=scheme)
        encoder.train()
        encoder(*_batch()).sum().backward()
        unused = [
            name for name, parameter in encoder.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not unused, f'{scheme}: {len(unused)} parameter(s) unused: {unused[:6]}'


def test_the_rotary_arm_holds_its_phase_at_production_scale():
    """The float64 reduction, checked where float32 alone would fail.

    Against the fastest band of the value ladder, `t / lambda` reaches ~4e5 radians at the far end
    of the history window, where float32 resolves a phase only to about 0.06 rad. Taking the
    remainder in float64 first keeps the cosine and sine arguments inside [0, 2*pi).
    """
    rotary = RotaryTemporalEncoding(N_HEADS, D_MODEL // N_HEADS, VALUE_P_MIN, VALUE_P_MAX)
    generator = torch.Generator().manual_seed(6)
    q = torch.randn(1, N_HEADS, 4, D_MODEL // N_HEADS, generator=generator)
    k = torch.randn(1, N_HEADS, 4, D_MODEL // N_HEADS, generator=generator)

    near = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    far = near + VALUE_GAP_MAX

    def score(positions):
        rotated_q, rotated_k = rotary(q, k, positions)
        return rotated_q @ rotated_k.transpose(-2, -1)

    delta = (score(near) - score(far)).abs().max().item()
    scale = score(near).abs().max().item()
    print(f'  phase drift at t = {VALUE_GAP_MAX:.0f} h: {delta:.2e} on a score of {scale:.1f}')
    assert delta / scale < 1e-3, f'phase was lost at production timestamps ({delta:.3e})'





# --------------------------------------------------------------------------------------------
# Config threading
# --------------------------------------------------------------------------------------------

ENTRY_POINTS = (
    'run_experiment_accelerate.py',
    'tune_hyperparameters_accelerate.py',
    'dump_finetuned_predictions.py',
    'TransEHR2/test_tune_hyperparameters.py',
)
LADDER_KEYS = ('POSITION_ENCODING', 'VALUE_LADDER_P_MIN', 'VALUE_LADDER_P_MAX',
               'EVENT_LADDER_P_MIN', 'EVENT_LADDER_P_MAX')


def _encoder_call_sites():
    """Every ValueDataEncoder/EventDataEncoder construction in the entry-point scripts.

    Parsed rather than imported: these modules pull in tensorboard and accelerate, so importing
    them to check a keyword would make this probe depend on the training environment.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ENTRY_POINTS:
        path = root / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in ('ValueDataEncoder', 'EventDataEncoder'):
                yield name, node


def test_every_construction_site_takes_the_encoding_arm():
    """A site that misses the arm silently runs the additive encoding inside a rotary experiment.

    There are more of these than the plan counted -- the two predictor encoders are rebuilt twice
    in run_experiment_accelerate.py, and the tuner exists in two copies -- which is the reason to
    check by parsing rather than by remembering.
    """
    import ast

    sites = list(_encoder_call_sites())
    assert len(sites) >= 12, f'only found {len(sites)} construction sites; the scan is not working'

    missing = []
    mismatched = []
    for filename, node in sites:
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords}
        for required in ('position_encoding', 'ladder_p_min', 'ladder_p_max'):
            if required not in keywords:
                missing.append(f'{filename}:{node.lineno} {node.func.id} lacks {required}')

        # The value stream sees history; the event stream is in-stay only post fix 03. They have
        # different gap ranges, so they must not share a ladder.
        wanted = 'VALUE_LADDER' if node.func.id == 'ValueDataEncoder' else 'EVENT_LADDER'
        bounds = keywords.get('ladder_p_min', '')
        if 'LADDER' in bounds and wanted not in bounds:
            mismatched.append(f'{filename}:{node.lineno} {node.func.id} is given {bounds}')

    assert not missing, '\n'.join(missing)
    assert not mismatched, '\n'.join(mismatched)
    print(f'  {len(sites)} construction sites, all threaded')


def test_the_shipped_tuning_config_parses_to_numbers():
    """YAML 1.1 reads `7.9e6` as a string -- the exponent needs an explicit sign.

    A string reaches `build_frequency_ladder` and compares against a float, so the failure is a
    TypeError inside model construction with nothing pointing at the config file.
    """
    import pathlib

    import yaml

    path = (
        pathlib.Path(__file__).resolve().parent
        / 'configs' / 'experiments' / 'tune_hyperparameters.yaml'
    )
    config = yaml.safe_load(path.read_text())

    assert config['POSITION_ENCODING'] in ('additive', 'rope')
    for key in LADDER_KEYS[1:]:
        assert isinstance(config[key], float), (
            f'{key} parsed as {type(config[key]).__name__} ({config[key]!r}); '
            'yaml 1.1 needs a decimal point and a signed exponent'
        )
    assert config['VALUE_LADDER_P_MAX'] > config['EVENT_LADDER_P_MAX'], (
        'the value stream spans history and must reach further than the in-stay event stream'
    )
    # And the bounds actually build, which is the only check that exercises both together.
    build_frequency_ladder(D_MODEL // 2, config['VALUE_LADDER_P_MIN'], config['VALUE_LADDER_P_MAX'])
    build_frequency_ladder(D_MODEL // 2, config['EVENT_LADDER_P_MIN'], config['EVENT_LADDER_P_MAX'])


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
