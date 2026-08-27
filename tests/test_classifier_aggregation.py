"""
Tests for MixedClassifier's sequence aggregation, specifically the `aggr='none'` readout.

The extraction layout writes history right-justified in [0, H) and in-stay data
left-justified from index H. An episode with fewer historical records than the history
window therefore carries *leading* padding, so the non-padding block starts somewhere
after index 0. That makes the count of observed records unusable as an index into the
sequence: counting gives h + e - 1 while the final in-stay record sits at H + e - 1.

These tests pin the readout to the last non-padding timestep under that layout. They use a
stub encoder whose output at timestep t is the constant vector [t, ..., t], so the selected
timestep is readable straight off the model output once the classification head is replaced
with identities.
"""

import torch

from TransEHR2.models import MixedClassifier


# Extraction-time layout used throughout: H history slots, E in-stay slots.
H, E = 3, 4
D_ENC = 4


class _IndexEncoder(torch.nn.Module):
    """Stub encoder emitting the constant vector [t, ..., t] at timestep t."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, indicators, times, masks):
        batch_size, n_timesteps = masks.shape
        idx = torch.arange(n_timesteps, dtype=torch.float32)
        return idx.view(1, n_timesteps, 1).expand(batch_size, n_timesteps, self.d_model).clone()


def _build_model() -> MixedClassifier:
    model = MixedClassifier(
        event_encoder=_IndexEncoder(D_ENC),
        val_encoder=None,
        d_event_enc=D_ENC,
        d_val_enc=0,
        d_statics=0,
        num_classes=D_ENC,
        aggr='none',
    )
    # Read the aggregated embedding directly rather than a learned projection of it.
    model.linear = torch.nn.Identity()
    model.linear1 = torch.nn.Identity()
    model.eval()
    return model


def _event_batch(masks: torch.Tensor):
    batch_size, n_timesteps = masks.shape
    return {
        'event_data': {
            'indicators': torch.zeros(batch_size, n_timesteps, 1),
            'times': torch.zeros(batch_size, n_timesteps),
            'masks': masks,
        }
    }


def _selected_timesteps(model, masks) -> torch.Tensor:
    with torch.no_grad():
        out = model(_event_batch(masks))
    # The head is the identity, so the output is gelu() of the selected constant vector.
    assert torch.allclose(out, out[:, [0]].expand_as(out)), 'stub encoder output is not constant across dims'
    return out[:, 0]


def _expected(indices) -> torch.Tensor:
    return torch.nn.functional.gelu(torch.tensor(indices, dtype=torch.float32))


def test_aggr_none_selects_final_in_stay_record_under_leading_padding():
    """A short history leaves leading padding; the readout must still take the last record."""
    masks = torch.zeros(2, H + E)
    masks[0, H - 1:H + 3] = 1.0  # 1 history record, 3 in-stay records -> last at H + 2
    masks[1, 0:H + 2] = 1.0      # full history, 2 in-stay records     -> last at H + 1

    selected = _selected_timesteps(_build_model(), masks)
    assert torch.allclose(selected, _expected([H + 2, H + 1])), (
        f'expected timesteps [{H + 2}, {H + 1}], got {selected.tolist()}'
    )


def test_aggr_none_matches_record_count_when_history_is_full():
    """With no leading padding the count of observed records is a valid index; agree with it."""
    masks = torch.zeros(3, H + E)
    for row, n_episode_records in enumerate((1, 2, E)):
        masks[row, 0:H + n_episode_records] = 1.0

    selected = _selected_timesteps(_build_model(), masks)
    assert torch.allclose(selected, _expected([H, H + 1, H + E - 1])), (
        f'expected timesteps [{H}, {H + 1}, {H + E - 1}], got {selected.tolist()}'
    )


def test_aggr_none_yields_zero_for_all_padding_rows():
    """Episodes with no observed records contribute a zero vector, as the masking implies."""
    masks = torch.zeros(2, H + E)
    masks[1, H - 1:H + 2] = 1.0

    selected = _selected_timesteps(_build_model(), masks)
    assert torch.allclose(selected, _expected([0.0, H + 1])), (
        f'expected [0.0, gelu({H + 1})], got {selected.tolist()}'
    )


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\n{len(tests)} tests passed')
