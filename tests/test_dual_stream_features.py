"""Probes for a feature carried by both the value and the event stream.

A feature may be named in `VALUED_FEATS` and `EVENT_FEATS` at once, in which case its
magnitudes reach the value encoder and its occurrences reach the Hawkes process. Nothing in
the pipeline partitions the two lists -- the reader resolves columns by name independently per
list and `_get_tensor_dimensions` sizes each list on its own -- but that is a property worth
pinning rather than rediscovering, because the alternative failure is silent: a feature would
lose either its values or its arrival times with no error raised.

The ordinal case is separate from the numeric one. An ordinal keeps its levels on the value
side, where it occupies `size` one-hot columns, and collapses to a single occurrence column on
the event side.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransEHR2.data.preprocessing import DataProcessor, _get_tensor_dimensions


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIABLE_PROPERTIES = os.path.join(REPO, 'data', 'variable_properties.yaml')

# 'Lactate' is numeric and 'GCS eye opening' ordinal; both are event features today and both
# are given to the value stream as well.
VALUED = ['Heart rate', 'Lactate', 'GCS eye opening']
EVENT = ['Lactate', 'GCS eye opening', 'Serum creatinine']
STATIC = ['Age', 'Gender']

HOURS = [-10.0, -5.0, 0.0]


def build():
    """A processor over the overlapping lists, and a frame of three timesteps."""
    dims = _get_tensor_dimensions(VARIABLE_PROPERTIES, VALUED, EVENT, [], STATIC,
                                  max_ts_len=len(HOURS), n_episodes=1, phenotype_dim=3)
    processor = DataProcessor(VARIABLE_PROPERTIES, VALUED, EVENT, [], STATIC, dims)
    frame = pd.DataFrame(
        {
            'Heart rate': [80.0, np.nan, 90.0],
            'Lactate': [1.5, 2.5, np.nan],
            'GCS eye opening': [3.0, np.nan, 2.0],
            'Serum creatinine': [np.nan, 1.1, np.nan],
        },
        index=pd.to_timedelta(HOURS, unit='h'),
    )
    return processor, dims, frame


def test_dimensions_count_an_overlapping_feature_in_both_streams():
    _, dims, _ = build()
    # Heart rate and Lactate are the numeric valued features; GCS eye opening is the ordinal.
    assert dims.n_numeric_feats == 2
    assert dims.n_ordinal_feats == 1
    # The event stream keeps all three of its own, overlap included.
    assert dims.n_event_feats == len(EVENT)


def test_a_numeric_feature_in_both_lists_keeps_its_values_and_its_arrivals():
    processor, _, frame = build()

    (_, numeric_indicators, numeric_values, *_) = processor.process_valued_data(frame[VALUED])
    column = processor.numeric_feats.index('Lactate')
    values = np.asarray(numeric_values[column]).ravel()
    # Charted at the first two timesteps and missing at the third.
    assert list(numeric_indicators[:, column]) == [1.0, 1.0, 0.0]
    assert values[0] == 1.5 and values[1] == 2.5

    _, event_indicators = processor.process_event_data(frame[EVENT])
    event_column = EVENT.index('Lactate')
    assert list(event_indicators[:, event_column]) == [1.0, 1.0, 0.0]


def test_an_ordinal_in_both_lists_keeps_its_levels_only_on_the_value_side():
    processor, dims, frame = build()

    out = processor.process_valued_data(frame[VALUED])
    ordinal_indicators, ordinal_values = out[5], out[6]
    assert list(ordinal_indicators[:, 0]) == [1.0, 0.0, 1.0]

    # One block of `size` one-hot columns per timestep, so distinct levels are distinguishable.
    size = dims.ordinal_feat_dims[0]
    blocks = np.asarray(ordinal_values[0]).reshape(len(HOURS), size)
    assert blocks[0].sum() == 1 and blocks[2].sum() == 1
    assert blocks[0].argmax() != blocks[2].argmax(), (
        'a GCS of 3 and a GCS of 2 landed on the same one-hot column, so the value stream '
        'is not carrying the level'
    )
    assert blocks[1].sum() == 0, 'an unobserved timestep must set no level'

    # The event side records that a measurement happened and nothing about which level.
    _, event_indicators = processor.process_event_data(frame[EVENT])
    assert list(event_indicators[:, EVENT.index('GCS eye opening')]) == [1.0, 0.0, 1.0]


def test_event_column_order_follows_its_own_list():
    """Overlap must not reorder the event stream, whose columns are keyed by position."""
    processor, _, frame = build()
    _, event_indicators = processor.process_event_data(frame[EVENT])
    # Serum creatinine is charted at the middle timestep alone, and is last in EVENT.
    assert list(event_indicators[:, EVENT.index('Serum creatinine')]) == [0.0, 1.0, 0.0]
    assert event_indicators.shape[1] == len(EVENT)


def test_the_two_streams_agree_on_timestamps():
    """Both streams are read off one frame, so an overlapping feature's arrival times match."""
    processor, _, frame = build()
    value_times, *_ = processor.process_valued_data(frame[VALUED])
    event_times, _ = processor.process_event_data(frame[EVENT])
    assert list(value_times) == list(event_times) == HOURS
