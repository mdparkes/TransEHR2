"""Probes for the value-stream type partition, and the integrity check's expected-empty list.

`partition_valued_feats` recovers what each indicator column holds by repeating the grouping
extraction performs. The data integrity check reports which features carry no data, so a wrong
name there makes it worse than useless: it names an innocent feature and clears the guilty one.

The expected-empty list exists so that a feature MIMIC-IV genuinely does not carry does not
hold the check permanently red, which is how a check stops being read.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_data_integrity as integrity
from TransEHR2.data.preprocessing import VALUE_TYPES, partition_valued_feats


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPERTIES = os.path.join(REPO, 'data', 'variable_properties.yaml')
CONFIG = os.path.join(REPO, 'TransEHR2', 'configs', 'datasets', 'mimic4.yaml')


@pytest.fixture(scope='module')
def properties():
    with open(PROPERTIES) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope='module')
def valued_feats():
    with open(CONFIG) as handle:
        return yaml.safe_load(handle)['VALUED_FEATS']


def test_the_partition_groups_by_type_not_by_position(properties, valued_feats):
    """Slicing the config list by per-type counts misnames most columns.

    VALUED_FEATS is alphabetized, so the ordinals sit in the middle of it. Taking the first
    n_numeric entries as the numeric names therefore picks up the ordinals and drops the last
    numerics -- which is what the check used to do.
    """
    grouped = partition_valued_feats(valued_feats, properties)
    numeric = grouped['numeric']

    by_position = valued_feats[:len(numeric)]
    mismatched = [index for index in range(len(numeric)) if by_position[index] != numeric[index]]
    assert mismatched, 'the config list happens to be grouped by type; this probe is void'
    assert numeric[mismatched[0]] != by_position[mismatched[0]]
    # Every ordinal must be absent from the numeric names.
    assert not set(grouped['ordinal']) & set(numeric)


def test_every_valued_feature_lands_in_exactly_one_stream(properties, valued_feats):
    grouped = partition_valued_feats(valued_feats, properties)
    placed = [name for stream in VALUE_TYPES for name in grouped[stream]]
    assert sorted(placed) == sorted(valued_feats)
    assert len(placed) == len(set(placed))


def test_column_order_within_a_type_follows_the_config(properties, valued_feats):
    """The extraction preserves config order within a type, so the names must too."""
    grouped = partition_valued_feats(valued_feats, properties)
    for stream in VALUE_TYPES:
        expected = [name for name in valued_feats
                    if properties[name]['type'] == stream]
        assert grouped[stream] == expected


def test_an_undeclared_feature_is_refused(properties):
    with pytest.raises(ValueError, match='not in the variable properties'):
        partition_valued_feats(['Heart rate', 'Nonexistent'], properties)


def test_a_type_with_no_indicator_tensor_is_refused(properties):
    """A text feature in VALUED_FEATS occupies no value indicator and must not be silent."""
    with pytest.raises(ValueError, match='occupies no value indicator tensor'):
        partition_valued_feats(['Discharge Summary'], properties)


def test_expected_empty_features_are_declared_features(properties, valued_feats):
    """A name in the list that no config declares is a stale entry hiding nothing."""
    declared = set(valued_feats) | set(properties)
    unknown = integrity.EXPECTED_EMPTY - declared
    assert not unknown, f'EXPECTED_EMPTY names features nothing declares: {sorted(unknown)}'
