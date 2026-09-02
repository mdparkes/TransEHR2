"""Probes for the generator loss spike report.

The generator's per-epoch mean is dominated by a few steps orders of magnitude above the median,
while the median sits near the validation mean. Numeric features are the only term scored by
squared error, so a single large residual can do that -- and the residual is either a target the
standardization left in the tail or a prediction the model produced. The report exists to tell
those apart, so what it must get right is which magnitudes it attributes to which side.
"""

import re

import pytest
import torch

from TransEHR2.utils import describe_loss_spike


BATCH, STEPS, DIM = 4, 6, 3


def build(pred_extreme, target_extreme, feature_type='numeric'):
    """One feature of `feature_type`, with a planted extreme on each side."""
    values_key = 'embedded_values' if feature_type == 'text' else 'values'

    prediction = torch.full((BATCH, STEPS, DIM), 0.5)
    prediction[0, 0, 0] = pred_extreme

    value_mask = torch.zeros(BATCH, STEPS, DIM)
    value_mask[0, 0, :] = 1.0
    value_mask[1, 1, :] = 1.0
    indicator_mask = torch.zeros(BATCH, STEPS, 1)
    indicator_mask[0, 0, 0] = 1.0
    indicator_mask[1, 1, 0] = 1.0

    # Targets arrive already reduced to the masked positions.
    target = torch.full((2, DIM), 0.4)
    target[0, 0] = target_extreme

    return (
        {feature_type: {values_key: [prediction]}},
        {feature_type: {values_key: [target]}},
        {feature_type: {'values': [value_mask], 'indicators': indicator_mask}},
    )


def row(report, feature_type):
    """The (masked, max|pred|, max|target|) triple the report prints for a feature type."""
    line = next(l for l in report.splitlines() if l.strip().startswith(feature_type))
    numbers = re.findall(r'[\d,]+\.?\d*', line.replace(feature_type, '', 1))
    return [float(n.replace(',', '')) for n in numbers]


def test_a_runaway_prediction_is_attributed_to_the_prediction():
    predictions, targets, masks = build(pred_extreme=9_000.0, target_extreme=0.4)
    report = describe_loss_spike(predictions, targets, masks, gen_loss=1422.0, reference=6.0)
    masked, worst_pred, worst_target = row(report, 'numeric')
    assert worst_pred == pytest.approx(9_000.0)
    assert worst_target < 10, 'a modest target must not be reported as extreme'


def test_a_runaway_target_is_attributed_to_the_target():
    predictions, targets, masks = build(pred_extreme=0.5, target_extreme=84_000.0)
    report = describe_loss_spike(predictions, targets, masks, gen_loss=1422.0, reference=6.0)
    masked, worst_pred, worst_target = row(report, 'numeric')
    assert worst_target == pytest.approx(84_000.0)
    assert worst_pred < 10, 'a modest prediction must not be reported as extreme'


def test_the_masked_count_is_the_number_of_scored_entries():
    predictions, targets, masks = build(pred_extreme=1.0, target_extreme=1.0)
    report = describe_loss_spike(predictions, targets, masks, gen_loss=100.0, reference=1.0)
    masked, _, _ = row(report, 'numeric')
    assert masked == 2 * DIM, 'two masked positions of DIM components each'


def test_the_prediction_extreme_ignores_unmasked_positions():
    """The loss only scores masked entries, so a large value elsewhere is not the cause."""
    predictions, targets, masks = build(pred_extreme=0.5, target_extreme=0.4)
    predictions['numeric']['values'][0][3, 5, 2] = 50_000.0   # not in the mask
    report = describe_loss_spike(predictions, targets, masks, gen_loss=100.0, reference=1.0)
    _, worst_pred, _ = row(report, 'numeric')
    assert worst_pred < 10, 'an unmasked outlier was reported as if the loss had scored it'


def test_text_is_indexed_by_the_indicator_mask():
    """Text is masked whole-record and indexes feature_masks[:, :, f], not a value mask -- the
    same asymmetry the loss itself has."""
    predictions, targets, masks = build(pred_extreme=7.0, target_extreme=0.4,
                                        feature_type='text')
    report = describe_loss_spike(predictions, targets, masks, gen_loss=100.0, reference=1.0)
    _, worst_pred, _ = row(report, 'text')
    assert worst_pred == pytest.approx(7.0)


def test_the_ratio_to_the_reference_is_reported():
    predictions, targets, masks = build(pred_extreme=1.0, target_extreme=1.0)
    report = describe_loss_spike(predictions, targets, masks, gen_loss=1422.0, reference=6.0)
    assert '237x' in report


def test_a_feature_type_with_nothing_masked_is_omitted():
    """An empty target means the loss skipped it, so the report has nothing to say about it."""
    predictions, targets, masks = build(pred_extreme=1.0, target_extreme=1.0)
    targets['numeric']['values'] = [torch.zeros(0, DIM)]
    report = describe_loss_spike(predictions, targets, masks, gen_loss=100.0, reference=1.0)
    assert not any(l.strip().startswith('numeric') for l in report.splitlines())


def test_a_missing_feature_type_does_not_raise():
    """Zero-feature types are real: VALUED_FEATS carries no ordinal features."""
    report = describe_loss_spike({}, {}, {}, gen_loss=100.0, reference=1.0)
    assert 'generator loss' in report
