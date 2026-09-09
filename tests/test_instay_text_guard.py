"""Probes for the exclusion of text recorded at or after ICU admission.

Every text feature is a discharge-time artifact of an admission: a discharge summary and the
diagnosis list that goes with it. Extraction keeps whatever falls inside the episode window,
so a stay shorter than that window contributes its own discharge documentation as an in-stay
record -- and that document states the outcome being predicted. `collate_tensorized` drops it
whatever the record switches say.

The exclusion has two halves and only one of them is visible. Clearing the indicator is not
enough: embeddings travel in a separate sparse block keyed by timestep, so a record cleared
from the indicator still reaches the value encoder through its embedding. Both halves are
checked here, and the dense tensor is checked as well, because that is what the encoder reads.

History is located by position, matching `TransEHR2.data.cohorts` and the rest of the collate
function, so `history_len_steps` is the boundary and a batch collated without a history region
carries no text at all.
"""

import numpy as np
import pytest
import torch

from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import densify_text_embeddings


TS_LEN, HIST, EMBED_DIM, N_TEXT = 8, 5, 4, 2
SUMMARY, DIAGNOSIS = 0, 1


def episode(text_at, numeric_at=(), ts_len=TS_LEN, n_text=N_TEXT):
    """One episode dict, with a text record of each named feature at the given timesteps.

    Args:
        text_at: Iterable of (timestep, feature index) carrying a text record.
        numeric_at: Timesteps carrying a numeric record.
        ts_len: Length of the timestep axis.
        n_text: Number of text features.

    Returns:
        The dict `MixedDataset.__getitem__` emits, with the embedding of the note at timestep
        `t` of feature `f` set to `t + 1` in every component so it can be traced.
    """
    text_indicators = torch.zeros(ts_len, n_text)
    numeric_indicators = torch.zeros(ts_len, 1)
    masks = torch.zeros(ts_len)
    for step, feature in text_at:
        text_indicators[step, feature] = 1.0
        masks[step] = 1.0
    for step in numeric_at:
        numeric_indicators[step, 0] = 1.0
        masks[step] = 1.0

    embeddings = []
    for feature in range(n_text):
        steps = sorted(step for step, f in text_at if f == feature)
        embeddings.append((
            torch.tensor(steps, dtype=torch.int64),
            torch.stack([torch.full((EMBED_DIM,), float(step + 1)) for step in steps])
            if steps else torch.zeros(0, EMBED_DIM),
        ))

    return {
        'val_numeric_indicators': numeric_indicators,
        'val_numeric_values': [torch.zeros(ts_len, 1)],
        'val_categorical_indicators': torch.zeros(ts_len, 0),
        'val_categorical_values': [],
        'val_ordinal_indicators': torch.zeros(ts_len, 0),
        'val_ordinal_values': [],
        'val_multilabel_indicators': torch.zeros(ts_len, 0),
        'val_multilabel_values': [],
        'val_text_indicators': text_indicators,
        'val_text_embeddings': embeddings,
        'val_times': torch.arange(ts_len, dtype=torch.float32) - HIST,
        'val_masks': masks,
        'event_indicators': torch.zeros(ts_len, 1),
        'event_times': torch.arange(ts_len, dtype=torch.float32) - HIST,
        'event_masks': torch.ones(ts_len),
        'static_data': torch.zeros(2),
        'mortality': torch.tensor(0.0),
        'length_of_stay': torch.tensor(0.0),
        'phenotype': torch.zeros(1),
    }


def kept_timesteps(collated, feature):
    """Timesteps whose embedding for `feature` survived collation."""
    block = collated['val_data']['text']['sparse_embeddings'][feature]
    return sorted(int(step) for step in block['timestep_index'])


@pytest.mark.parametrize('feature,name', [(SUMMARY, 'discharge summary'),
                                          (DIAGNOSIS, 'diagnosis descriptions')])
@pytest.mark.parametrize('step', list(range(HIST, TS_LEN)))
def test_a_text_record_at_or_after_admission_is_dropped(feature, name, step):
    """Both text features and every in-stay position, since a stay shorter than the episode
    window can place its discharge documentation anywhere inside it."""
    collated = collate_tensorized([episode([(step, feature)])], history_len_steps=HIST)
    indicators = collated['val_data']['text']['indicators']
    assert indicators[0, step, feature] == 0.0, f'{name} indicator survived at {step}'
    assert step not in kept_timesteps(collated, feature), \
        f'{name} embedding survived at {step}'


@pytest.mark.parametrize('feature', [SUMMARY, DIAGNOSIS])
def test_a_pre_admission_text_record_survives(feature):
    """The guard must not reach into the history region, which is the text the model reads."""
    collated = collate_tensorized([episode([(HIST - 1, feature)])], history_len_steps=HIST)
    assert collated['val_data']['text']['indicators'][0, HIST - 1, feature] == 1.0
    assert kept_timesteps(collated, feature) == [HIST - 1]


@pytest.mark.parametrize('historical', [True, False])
@pytest.mark.parametrize('instay', [True, False])
def test_no_switch_combination_admits_in_stay_text(historical, instay):
    """The exclusion is not a switch. USE_INSTAY_RECORDS keeps in-stay non-text records, and
    must not carry the text in with them."""
    collated = collate_tensorized(
        [episode([(HIST, SUMMARY), (HIST + 2, DIAGNOSIS)], numeric_at=(HIST, HIST + 2))],
        use_historical_text_records=historical,
        use_instay_records=instay,
        history_len_steps=HIST,
    )
    assert collated['val_data']['text']['indicators'][0, HIST:].sum() == 0.0
    for feature in (SUMMARY, DIAGNOSIS):
        assert kept_timesteps(collated, feature) == []


def test_the_dense_tensor_the_encoder_reads_holds_no_in_stay_note():
    """The sparse block is an intermediate form; this is what the value encoder receives."""
    collated = collate_tensorized(
        [episode([(HIST - 2, SUMMARY), (HIST + 1, SUMMARY), (HIST + 2, DIAGNOSIS)])],
        history_len_steps=HIST,
    )
    dense = densify_text_embeddings(collated)['val_data']['text']['embedded_values']
    assert dense[0, HIST:].abs().sum() == 0.0, 'an in-stay embedding reached the dense tensor'
    # The surviving note carries its own timestep in every component, so this also checks that
    # the guard did not shift the rows it kept.
    assert torch.equal(dense[0, HIST - 2, SUMMARY],
                       torch.full((EMBED_DIM,), float(HIST - 1)))


def test_a_timestep_left_with_nothing_becomes_padding():
    """A note at a timestep of its own is the whole of that timestep, so clearing it must
    clear the mask too rather than leave an observed step with no record in it."""
    collated = collate_tensorized([episode([(HIST + 1, SUMMARY)])], history_len_steps=HIST)
    assert collated['val_data']['masks'][0, HIST + 1] == 0.0


def test_a_timestep_keeps_its_mask_when_another_record_remains():
    """Dropping the note must not drop the numeric record sharing its timestep."""
    collated = collate_tensorized([episode([(HIST + 1, SUMMARY)], numeric_at=(HIST + 1,))],
                                  history_len_steps=HIST)
    assert collated['val_data']['masks'][0, HIST + 1] == 1.0
    assert collated['val_data']['numeric']['indicators'][0, HIST + 1, 0] == 1.0


def test_without_a_history_region_no_text_survives():
    """HISTORY_LEN_STEPS of 0 leaves every timestep at or after admission, so there is no
    position a text record could occupy and still be pre-admission."""
    collated = collate_tensorized([episode([(0, SUMMARY), (3, DIAGNOSIS)])],
                                  history_len_steps=0)
    assert collated['val_data']['text']['indicators'].sum() == 0.0
    for feature in (SUMMARY, DIAGNOSIS):
        assert kept_timesteps(collated, feature) == []


def test_a_batch_with_no_in_stay_text_is_left_alone():
    """The guard is a no-op where it has nothing to do, so it cannot perturb a run whose text
    is entirely pre-admission."""
    batch = [episode([(1, SUMMARY), (HIST - 1, DIAGNOSIS)], numeric_at=(HIST, HIST + 1))]
    expected_masks = batch[0]['val_masks'].clone()
    collated = collate_tensorized(batch, history_len_steps=HIST)
    assert torch.equal(collated['val_data']['masks'][0], expected_masks)
    assert kept_timesteps(collated, SUMMARY) == [1]
    assert kept_timesteps(collated, DIAGNOSIS) == [HIST - 1]
