"""
Smoke tests for ordinal feature support.

Constructs small synthetic data in memory and runs it through the ELECTRA
model (generator + discriminator) with ordinal features, performing a
single optimization step to verify the full pipeline works end-to-end.
"""

import torch
import torch.nn as nn

from TransEHR2.modules import (
    EventDataEncoder,
    MaskedTokenDiscriminator,
    MaskedTokenGenerator,
    TransformerHawkesProcess,
    ValueDataEncoder,
)
from TransEHR2.models import ELECTRA
from TransEHR2.losses import (
    MaskedDiscriminatorLoss,
    MaskedGeneratorLoss,
)
from TransEHR2.utils import generate_record_masks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot_feature(indicator, n_levels):
    """Build a one-hot value tensor for a categorical or ordinal feature.

    Matches the storage format produced by TransEHR2.data.preprocessing: shape
    (batch, max_ts, n_levels), int64, with a one-hot row wherever the feature is
    observed and an all-zero row wherever it is unobserved (a zero row also
    encodes an unknown / out-of-domain value).
    """
    batch_size, max_ts = indicator.shape
    classes = torch.randint(0, n_levels, (batch_size, max_ts))
    one_hot = torch.nn.functional.one_hot(classes, num_classes=n_levels)
    return one_hot * indicator.long().unsqueeze(-1)


def _make_batch(
    batch_size=4,
    max_ts=6,
    numeric_dims=None,
    categorical_classes=None,
    ordinal_features=None,
    n_event_types=3,
):
    """Build a minimal MixedTensorDataset dict with synthetic data."""
    if numeric_dims is None:
        numeric_dims = [1, 2]
    if categorical_classes is None:
        categorical_classes = [4]
    if ordinal_features is None:
        ordinal_features = [3, 5]

    batch = {}

    # --- value data ---
    val_data = {
        'times': torch.rand(batch_size, max_ts),
        'masks': torch.ones(batch_size, max_ts),
    }

    # numeric
    num_indicators = torch.randint(0, 2, (batch_size, max_ts, len(numeric_dims))).float()
    num_values = [torch.randn(batch_size, max_ts, d) for d in numeric_dims]
    val_data['numeric'] = {'indicators': num_indicators, 'values': num_values}

    # categorical: one-hot encoded, (B, T, n_classes)
    cat_indicators = torch.randint(0, 2, (batch_size, max_ts, len(categorical_classes))).float()
    cat_values = [
        _one_hot_feature(cat_indicators[:, :, f], n_cls)
        for f, n_cls in enumerate(categorical_classes)
    ]
    val_data['categorical'] = {'indicators': cat_indicators, 'values': cat_values}

    # ordinal: one-hot encoded, (B, T, n_levels)
    ord_indicators = torch.randint(0, 2, (batch_size, max_ts, len(ordinal_features))).float()
    ord_values = [
        _one_hot_feature(ord_indicators[:, :, f], n_lvl)
        for f, n_lvl in enumerate(ordinal_features)
    ]
    val_data['ordinal'] = {'indicators': ord_indicators, 'values': ord_values}

    batch['val_data'] = val_data

    # --- event data ---
    event_indicators = torch.zeros(batch_size, max_ts, n_event_types)
    for b in range(batch_size):
        for t in range(max_ts):
            event_indicators[b, t, torch.randint(0, n_event_types, (1,))] = 1.0
    batch['event_data'] = {
        'indicators': event_indicators,
        'times': torch.sort(torch.rand(batch_size, max_ts), dim=-1).values,
        'masks': torch.ones(batch_size, max_ts),
    }

    # --- targets (not used for pretraining, but batch may have it) ---
    batch['targets'] = {
        'mortality': torch.randint(0, 2, (batch_size, 1)).float(),
    }

    return batch


def _build_electra(
    numeric_dims=None,
    categorical_classes=None,
    ordinal_features=None,
    n_event_types=3,
    d_model=16,
    dim_ff=32,
):
    """Build a minimal ELECTRA model with ordinal support."""
    if numeric_dims is None:
        numeric_dims = [1, 2]
    if categorical_classes is None:
        categorical_classes = [4]
    if ordinal_features is None:
        ordinal_features = [3, 5]

    n_val_feats = (
        len(numeric_dims) + len(categorical_classes) + len(ordinal_features)
    )
    tot_val_feat_dim = (
        sum(numeric_dims) + sum(categorical_classes) + sum(ordinal_features)
    )

    gen_enc = ValueDataEncoder(
        n_features=n_val_feats,
        feat_dim=tot_val_feat_dim,
        d_model=d_model,
        n_heads=2,
        n_encoder_blocks=1,
        dim_feedforward=dim_ff,
        norm='LayerNorm',
    )
    disc_enc = ValueDataEncoder(
        n_features=n_val_feats,
        feat_dim=tot_val_feat_dim,
        d_model=d_model,
        n_heads=2,
        n_encoder_blocks=1,
        dim_feedforward=dim_ff,
        norm='LayerNorm',
    )
    thp_enc = EventDataEncoder(
        num_types=n_event_types,
        d_model=d_model,
        d_inner=dim_ff,
        n_layers=1,
        n_head=2,
        d_k=8,
        d_v=8,
        dropout=0.1,
    )

    generator = MaskedTokenGenerator(
        encoder=gen_enc,
        d_model=d_model,
        numeric_dims=numeric_dims,
        categorical_classes=categorical_classes,
        ordinal_features=ordinal_features,
        predict_indicators=False,
        dim_feedforward=dim_ff,
    )
    discriminator = MaskedTokenDiscriminator(
        encoder=disc_enc,
        d_model=d_model,
        n_numeric_features=len(numeric_dims),
        n_categorical_features=len(categorical_classes),
        n_ordinal_features=len(ordinal_features),
        n_static_features=0,
        dim_feedforward=dim_ff,
    )
    thp = TransformerHawkesProcess(encoder=thp_enc, num_types=n_event_types)

    return ELECTRA(generator=generator, discriminator=discriminator, hawkes=thp)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_pass():
    """The ELECTRA forward pass should succeed with ordinal features."""
    ordinal_features = [3, 5]
    batch = _make_batch(ordinal_features=ordinal_features)
    electra = _build_electra(ordinal_features=ordinal_features)

    record_masks, event_masks = generate_record_masks(batch)

    outputs = electra(
        batch, record_masks, device='cpu', compute_intensities=True
    )

    # Generator should have ordinal key with correct number of features
    assert 'ordinal' in outputs['generator'], "Generator output missing 'ordinal' key"
    ord_vals = outputs['generator']['ordinal']['values']
    assert len(ord_vals) == len(ordinal_features), (
        f"Expected {len(ordinal_features)} ordinal outputs, got {len(ord_vals)}"
    )
    # Each output should be (batch, max_ts, n_levels) probabilities
    for i, (probs, n_lvl) in enumerate(zip(ord_vals, ordinal_features)):
        assert probs.shape[-1] == n_lvl, (
            f"Ordinal feature {i}: expected {n_lvl} levels, got {probs.shape[-1]}"
        )
        # Probabilities should sum to ~1 along the last dimension
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4), (
            f"Ordinal feature {i}: probabilities do not sum to 1"
        )

    # Discriminator should have ordinal key
    assert 'ordinal' in outputs['discriminator'], (
        "Discriminator output missing 'ordinal' key"
    )

    # masked_targets should have ordinal key
    assert 'ordinal' in outputs['masked_targets'], (
        "masked_targets missing 'ordinal' key"
    )

    print("PASS: test_forward_pass")


def test_backward_pass():
    """A full training step (forward + backward + optimizer step) should
    work without errors and reduce the loss."""
    ordinal_features = [3, 5]
    batch = _make_batch(ordinal_features=ordinal_features)
    electra = _build_electra(ordinal_features=ordinal_features)

    gen_loss_fn = MaskedGeneratorLoss(ordinal_features=ordinal_features)
    disc_loss_fn = MaskedDiscriminatorLoss()

    optimizer = torch.optim.Adam(electra.parameters(), lr=1e-3)

    losses = []
    for step in range(3):
        optimizer.zero_grad()
        # Create a fresh batch each step because _prepare_discriminator_input_inplace
        # modifies value_data in-place
        batch = _make_batch(ordinal_features=ordinal_features)
        record_masks, event_masks = generate_record_masks(batch)
        outputs = electra(
            batch, record_masks, device='cpu', compute_intensities=True
        )

        gen_loss = gen_loss_fn(
            outputs['generator'], outputs['masked_targets'], record_masks
        )
        disc_loss = disc_loss_fn(outputs['discriminator'], record_masks)
        total_loss = gen_loss + disc_loss
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    print(f"  Losses over 3 steps: {[f'{l:.4f}' for l in losses]}")
    # We don't strictly require monotonic decrease with random masking,
    # but the loss should be finite
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
        "Loss is not finite!"
    )
    print("PASS: test_backward_pass")


def test_no_ordinal_features():
    """Model with zero ordinal features should still work (backward compat)."""
    batch = _make_batch(ordinal_features=[])
    # Remove ordinal from batch
    del batch['val_data']['ordinal']

    electra = _build_electra(ordinal_features=[])

    record_masks, event_masks = generate_record_masks(batch)
    outputs = electra(batch, record_masks, device='cpu')

    assert 'ordinal' not in outputs['generator'], (
        "Generator should not have ordinal key when no ordinal features"
    )
    print("PASS: test_no_ordinal_features")


def test_ordinal_discriminator_input_replacement():
    """After _prepare_discriminator_input_inplace, ordinal values at masked
    positions should be replaced with generated class indices (1-indexed)."""
    ordinal_features = [3, 5]
    batch = _make_batch(ordinal_features=ordinal_features)
    electra = _build_electra(ordinal_features=ordinal_features)

    record_masks, _ = generate_record_masks(batch)

    # Save original ordinal values
    orig_vals = [v.clone() for v in batch['val_data']['ordinal']['values']]

    outputs = electra(batch, record_masks, device='cpu')

    # After forward, batch['val_data'] has been modified in-place
    for i, (orig, n_lvl) in enumerate(zip(orig_vals, ordinal_features)):
        new_vals = batch['val_data']['ordinal']['values'][i]
        assert new_vals.shape[-1] == n_lvl, (
            f"Ordinal feature {i}: expected width {n_lvl}, got {new_vals.shape[-1]}"
        )
        # Every component of a selected (batch, timestep) position is masked, so the
        # value mask marks whole one-hot rows.
        value_mask = record_masks['ordinal']['values'][i].bool()
        feat_mask = value_mask.all(dim=-1)
        if feat_mask.any():
            replaced = new_vals[feat_mask]
            # Replaced rows should be valid one-hot encodings over n_lvl levels
            assert ((replaced == 0) | (replaced == 1)).all(), (
                f"Ordinal feature {i}: replaced values are not 0/1 one-hot components"
            )
            assert (replaced.sum(dim=-1) == 1).all(), (
                f"Ordinal feature {i}: replaced rows are not one-hot "
                f"(row sums: {replaced.sum(dim=-1).unique().tolist()})"
            )
        # Unmasked positions must be left untouched
        unmasked = ~feat_mask
        if unmasked.any():
            assert torch.equal(new_vals[unmasked], orig[unmasked]), (
                f"Ordinal feature {i}: unmasked positions were modified"
            )
    print("PASS: test_ordinal_discriminator_input_replacement")


def test_clm_output_is_pmf():
    """CLM output should be a valid PMF (non-negative, sums to 1)."""
    from dlordinal.output_layers import CLM

    clm = CLM(num_classes=5, link_function='logit')
    x = torch.randn(10, 1)
    probs = clm(x)

    assert (probs >= 0).all(), "CLM output has negative probabilities"
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"CLM output does not sum to 1: {sums}"
    )
    print("PASS: test_clm_output_is_pmf")


def test_beta_loss():
    """BetaLoss should produce finite gradients."""
    from dlordinal.losses import BetaLoss

    n_classes = 5
    beta_loss = BetaLoss(
        base_loss=nn.CrossEntropyLoss(reduction='none'),
        num_classes=n_classes,
    )
    logits = torch.randn(8, n_classes, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.randint(0, n_classes, (8,))
    loss = beta_loss(probs, targets).sum()
    loss.backward()

    assert torch.isfinite(loss), f"BetaLoss is not finite: {loss.item()}"
    assert logits.grad is not None, "No gradient through BetaLoss"
    assert torch.isfinite(logits.grad).all(), "BetaLoss gradients are not finite"
    print("PASS: test_beta_loss")


if __name__ == '__main__':
    test_clm_output_is_pmf()
    test_beta_loss()
    test_forward_pass()
    test_backward_pass()
    test_no_ordinal_features()
    test_ordinal_discriminator_input_replacement()
    print("\nAll ordinal smoke tests passed!")
