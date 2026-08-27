"""End-to-end training probes for the encoder attention refactor, on simulated data.

The unit probes in ``test_encoder_refactor.py`` pin the arithmetic. These run the migrated
encoders inside the models that use them -- ELECTRA for pretraining, MixedClassifier for the
downstream task -- on a synthetic cohort with irregular timestamps and realistic padding, and
check that the whole thing still optimizes.

Three things are being asked:

* Does it train? Pretraining loss must fall and stay finite, including on episodes that are
  entirely padding, which is the case that used to produce NaN.
* Does it learn something that needs the time axis? The finetuning probe labels an episode by the
  *sign of its trend*, which no readout can recover from a bag of timesteps.
* Is the rotary seam usable end to end? The same runs are repeated with a stand-in rotation
  installed at the seam, forward and backward, to show a position transform trains rather than
  merely evaluating.

Run directly (``python -m TransEHR2.test_encoder_simulated_training``) or under pytest.
"""

import copy

import torch

from TransEHR2.losses import MaskedDiscriminatorLoss, MaskedGeneratorLoss, TransformerHawkesLoss
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import (
    EventDataEncoder,
    MaskedTokenDiscriminator,
    MaskedTokenGenerator,
    TransformerHawkesProcess,
    ValueDataEncoder,
)
from TransEHR2.utils import generate_record_masks


D_MODEL = 32
DIM_FF = 32
N_HEADS = 2
NUMERIC_DIMS = [1, 2]
N_EVENT_TYPES = 3
N_VAL_FEATURES = len(NUMERIC_DIMS)
TOTAL_FEAT_DIM = sum(NUMERIC_DIMS)


class _Rotator(torch.nn.Module):
    """A parameter-free, position-dependent rotation of q and k.

    Not RoPE -- every channel pair turns at the same rate rather than on a ladder of frequencies --
    but it has the shape of one, which is what the seam has to carry: no parameters, no state
    between calls, and a gradient path back through the encoder.
    """

    def forward(self, q, k, positions):
        angle = positions[:, None, :, None] * 0.25
        return self._rotate(q, angle), self._rotate(k, angle)

    @staticmethod
    def _rotate(x, angle):
        even, odd = x[..., 0::2], x[..., 1::2]
        cos, sin = torch.cos(angle), torch.sin(angle)
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


# --------------------------------------------------------------------------------------------
# Simulated cohort
# --------------------------------------------------------------------------------------------

def _simulate_cohort(n_episodes=24, max_steps=12, seed=0, all_padding_episode=True):
    """A batch of synthetic episodes in the layout the collate function produces.

    Every episode gets its own observed length and its own irregular hourly grid, so records land
    at different timestamps in different episodes -- which is what the temporal encoding is for.
    The label is the sign of the trend in the first numeric feature: values rise through the stay
    for half the cohort and fall for the other half, with the same marginal distribution either
    way, so nothing but the ordering separates the classes.
    """
    generator = torch.Generator().manual_seed(seed)

    lengths = torch.randint(4, max_steps + 1, (n_episodes,), generator=generator)
    labels = (torch.arange(n_episodes) % 2).float()

    masks = torch.zeros(n_episodes, max_steps)
    times = torch.zeros(n_episodes, max_steps)
    trend_values = torch.zeros(n_episodes, max_steps, 1)

    for episode in range(n_episodes):
        length = int(lengths[episode])
        masks[episode, :length] = 1.0
        gaps = 0.5 + torch.rand(length, generator=generator) * 1.5
        times[episode, :length] = gaps.cumsum(0)
        ramp = torch.linspace(-1.0, 1.0, length)
        direction = 1.0 if labels[episode] > 0.5 else -1.0
        trend_values[episode, :length, 0] = (
            direction * 2.0 * ramp + 0.3 * torch.randn(length, generator=generator)
        )

    if all_padding_episode:
        # One episode with nothing in it at all: the shape that NaNs any delegated attention.
        masks[-1] = 0.0
        times[-1] = 0.0
        trend_values[-1] = 0.0

    observed = masks.unsqueeze(-1)
    other_values = torch.randn(n_episodes, max_steps, NUMERIC_DIMS[1], generator=generator) * observed

    batch = {
        'val_data': {
            'times': times,
            'masks': masks,
            'numeric': {
                'indicators': masks.unsqueeze(-1).expand(-1, -1, N_VAL_FEATURES).contiguous(),
                'values': [trend_values, other_values],
            },
        },
        'event_data': {
            'indicators': (
                torch.nn.functional.one_hot(
                    torch.randint(0, N_EVENT_TYPES, (n_episodes, max_steps), generator=generator),
                    N_EVENT_TYPES,
                ).float() * observed
            ),
            'times': times,
            'masks': masks,
        },
        'targets': {'mortality': labels.unsqueeze(-1)},
    }
    return batch, labels


def _build_electra(transform=None):
    torch.manual_seed(0)
    common = dict(
        n_features=N_VAL_FEATURES, feat_dim=TOTAL_FEAT_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_encoder_blocks=2, dim_feedforward=DIM_FF, dropout=0.1, activation='gelu',
        norm='LayerNorm', normalize_before=True, query_key_transform=transform,
    )
    generator = MaskedTokenGenerator(
        encoder=ValueDataEncoder(**common), d_model=D_MODEL, numeric_dims=NUMERIC_DIMS,
        categorical_classes=[], dim_feedforward=DIM_FF,
    )
    discriminator = MaskedTokenDiscriminator(
        encoder=ValueDataEncoder(**common), d_model=D_MODEL,
        n_numeric_features=len(NUMERIC_DIMS), n_categorical_features=0, n_ordinal_features=0,
        n_multilabel_features=0, n_static_features=0, dim_feedforward=DIM_FF,
    )
    hawkes = TransformerHawkesProcess(
        encoder=EventDataEncoder(
            num_types=N_EVENT_TYPES, d_model=D_MODEL, d_inner=DIM_FF, n_layers=2, n_head=N_HEADS,
            d_k=D_MODEL // N_HEADS, d_v=D_MODEL // N_HEADS, dropout=0.1, normalize_before=True,
            query_key_transform=transform,
        ),
        num_types=N_EVENT_TYPES,
    )
    return ELECTRA(generator=generator, discriminator=discriminator, hawkes=hawkes)


def _build_classifier(transform=None):
    torch.manual_seed(0)
    value_encoder = ValueDataEncoder(
        n_features=N_VAL_FEATURES, feat_dim=TOTAL_FEAT_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_encoder_blocks=2, dim_feedforward=DIM_FF, dropout=0.1, activation='gelu',
        norm='LayerNorm', normalize_before=True, query_key_transform=transform,
    )
    event_encoder = EventDataEncoder(
        num_types=N_EVENT_TYPES, d_model=D_MODEL, d_inner=DIM_FF, n_layers=1, n_head=N_HEADS,
        d_k=D_MODEL // N_HEADS, d_v=D_MODEL // N_HEADS, dropout=0.1, normalize_before=True,
        query_key_transform=transform,
    )
    return MixedClassifier(
        event_encoder=event_encoder, val_encoder=value_encoder, d_event_enc=D_MODEL,
        d_val_enc=D_MODEL, d_statics=0, num_classes=1, aggr='mean',
    )


# --------------------------------------------------------------------------------------------
# Pretraining
# --------------------------------------------------------------------------------------------

def _run_pretraining(transform=None, steps=30, seed=0):
    """A short ELECTRA pretraining run over one fixed simulated cohort."""
    torch.manual_seed(seed)
    # No all-padding episode here: the Hawkes likelihood gates its base intensity on event index
    # 0, so an episode with no observed records is outside what the THP is defined on. The
    # encoders' own handling of that case is covered by test_encoder_refactor.py and by
    # test_classifier_output_is_finite_for_an_all_padding_episode below.
    batch, _ = _simulate_cohort(seed=seed, all_padding_episode=False)
    electra = _build_electra(transform)
    electra.train()

    generator_loss_fn = MaskedGeneratorLoss()
    discriminator_loss_fn = MaskedDiscriminatorLoss()
    hawkes_loss_fn = TransformerHawkesLoss()
    optimizer = torch.optim.Adam(electra.parameters(), lr=2e-3)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        # ELECTRA rewrites value_data in place, so each step gets its own copy of the cohort.
        step_batch = copy.deepcopy(batch)
        record_masks, _ = generate_record_masks(step_batch)
        outputs = electra(step_batch, record_masks, device='cpu', compute_intensities=True)
        intensities = outputs['thp_intensities']
        type_predictions, time_predictions = outputs['hawkes_predictions']
        hawkes_loss, _ = hawkes_loss_fn(
            intensities['obs_initial'],
            intensities['obs_conditional'],
            intensities['sampled'],
            step_batch['event_data'],
            type_predictions,
            time_predictions,
        )
        # The same three terms routines_accelerate.pretrain sums, so every encoder in the model
        # is attached to the objective -- the event encoder reaches it only through this one.
        loss = (
            generator_loss_fn(outputs['generator'], outputs['masked_targets'], record_masks)
            + discriminator_loss_fn(outputs['discriminator'], record_masks)
            + hawkes_loss
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return electra, losses


def test_electra_pretrains_on_simulated_data():
    """Pretraining must run, stay finite and come down."""
    _, losses = _run_pretraining()

    assert all(torch.isfinite(torch.tensor(value)) for value in losses), (
        f'pretraining loss went non-finite: {losses}'
    )
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first, f'pretraining loss did not fall: {first:.4f} -> {last:.4f}'
    print(f'  pretraining loss {first:.4f} -> {last:.4f} over {len(losses)} steps')


def test_pretraining_gradients_reach_every_encoder_parameter():
    """Both encoders have to be fully connected to the pretraining objective.

    Under DDP an unused parameter raises at the reduction step rather than being ignored, so a
    block that quietly falls out of the graph is a distributed-only failure.
    """
    electra, _ = _run_pretraining(steps=1)
    encoders = {
        'generator.encoder': electra.generator.encoder,
        'discriminator.encoder': electra.discriminator.encoder,
        'hawkes.encoder': electra.hawkes.encoder,
    }
    for name, encoder in encoders.items():
        unused = [
            f'{name}.{parameter_name}'
            for parameter_name, parameter in encoder.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not unused, f'{len(unused)} parameter(s) never received a gradient: {unused[:6]}'


def test_pretraining_runs_with_a_transform_at_the_rotary_seam():
    """A position transform must survive a real optimization loop, not just a forward pass."""
    _, plain = _run_pretraining()
    _, rotated = _run_pretraining(transform=_Rotator())

    assert all(torch.isfinite(torch.tensor(value)) for value in rotated), (
        f'pretraining with a transform installed went non-finite: {rotated}'
    )
    assert sum(rotated[-5:]) / 5 < sum(rotated[:5]) / 5, 'the transformed run did not train'
    assert abs(plain[0] - rotated[0]) > 1e-6, (
        'installing the transform changed nothing -- it is not reaching the attention'
    )
    print(f'  with rotation at the seam: {sum(rotated[:5]) / 5:.4f} -> {sum(rotated[-5:]) / 5:.4f}')


# --------------------------------------------------------------------------------------------
# Finetuning
# --------------------------------------------------------------------------------------------

def _run_finetuning(transform=None, steps=250, seed=0):
    """Fit the trend-sign task on one fixed simulated cohort."""
    torch.manual_seed(seed)
    batch, labels = _simulate_cohort(n_episodes=32, seed=seed, all_padding_episode=False)
    model = _build_classifier(transform)
    model.train()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(copy.deepcopy(batch)).squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        accuracy = (
            (model(copy.deepcopy(batch)).squeeze(-1) > 0).float() == labels
        ).float().mean().item()
    return losses, accuracy


def test_classifier_learns_a_task_that_needs_the_time_axis():
    """The label is the sign of the trend, so a bag of timesteps cannot separate the classes.

    Both classes draw from the same marginal distribution of values; only the order differs. An
    encoder whose attention ran across the batch instead of across time -- the bug this stack was
    rebuilt on top of -- cannot beat chance here.
    """
    losses, accuracy = _run_finetuning()

    assert all(torch.isfinite(torch.tensor(value)) for value in losses), 'finetuning loss went non-finite'
    assert accuracy > 0.85, f'trend-sign accuracy only reached {accuracy:.2f}'
    print(f'  trend-sign loss {losses[0]:.4f} -> {losses[-1]:.4f}, accuracy {accuracy:.2f}')


def test_classifier_output_is_finite_for_an_all_padding_episode():
    """An episode with no records must not poison the rest of the batch.

    NaN * 0 = NaN, so a NaN at a padded position survives the readout's padding multiply and then
    ``torch.sum`` spreads it across every prediction in the batch.
    """
    batch, _ = _simulate_cohort(n_episodes=8, seed=3, all_padding_episode=True)
    model = _build_classifier()
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    assert torch.isfinite(logits).all(), f'an all-padding episode produced {logits.flatten()}'


def test_finetuning_runs_with_a_transform_at_the_rotary_seam():
    losses, accuracy = _run_finetuning(transform=_Rotator())
    assert all(torch.isfinite(torch.tensor(value)) for value in losses), 'finetuning went non-finite'
    assert losses[-1] < losses[0], 'the transformed run did not train'
    print(f'  with rotation at the seam: loss {losses[0]:.4f} -> {losses[-1]:.4f}, accuracy {accuracy:.2f}')


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
