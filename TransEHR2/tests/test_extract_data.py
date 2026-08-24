"""End to end over B0's fixture root and the live config (section 4.4).

This is the test that pins the numbers: the 94 / 37 / 16 indicator widths
are counts of ``VALUED_FEATS`` entries by ``type`` in the real
``variable_properties.yaml`` (section A.3), so they are asserted against
the derived counts and never against literals. Everything the unit tests
check they check on a six-feature cohort they write themselves; this one
checks that the real pair of config files produces section 4.4's table.

Skipped when IBDdataprep's fixture has not been generated -- it is
untracked, and ``python -m IBDdataprep.make_fixture`` rebuilds it.
"""

import numpy as np
import pandas as pd
import pickle
import pytest
import subprocess
import sys
import yaml

from pathlib import Path

from extract_data import main as extract_main

# .../TransEHR2-IBDreadmission/TransEHR2/TransEHR2/tests/this_file.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IBDDATAPREP = PROJECT_ROOT / 'IBDdataprep'
FIXTURE_ROOT = IBDDATAPREP / 'IBDdataprep' / 'tests' / 'fixtures' / 'root'
DATASET_CONFIG = (PROJECT_ROOT / 'TransEHR2' / 'TransEHR2' / 'configs' /
                  'datasets' / 'RMT23345.yaml')
VARIABLE_PROPERTIES = (PROJECT_ROOT / 'data' / 'ibd' /
                       'variable_properties.yaml')
CLINVEC = PROJECT_ROOT / 'data' / 'resources' / 'ClinVec_atc.csv'

# The window is deliberately shorter than the fixture's longest episode,
# so the last-N filter actually bites.
MAX_EPISODE_LEN_STEPS = 3


@pytest.fixture(scope='module')
def extracted(tmp_path_factory):
    """Run the extractor over the fixture root with the live config."""
    for path in (FIXTURE_ROOT, VARIABLE_PROPERTIES, CLINVEC):
        if not path.exists():
            pytest.skip(
                f"{path} not found; run "
                f"'python -m IBDdataprep.make_fixture' in IBDdataprep/"
            )

    tmp_path = tmp_path_factory.mktemp('fixture_cohort')
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'root').symlink_to(FIXTURE_ROOT)

    build = subprocess.run(
        [sys.executable, '-m', 'IBDdataprep.build_labels',
         '-r', str(data_dir / 'root'), '-o', str(data_dir / 'labels.csv')],
        cwd=IBDDATAPREP, capture_output=True, text=True
    )
    if build.returncode != 0:
        pytest.skip(f"build_labels.py failed:\n{build.stderr}")

    config = yaml.safe_load(DATASET_CONFIG.read_text())
    config['DATA_DIR'] = str(data_dir)
    config['VARIABLE_PROPERTIES_PATH'] = str(VARIABLE_PROPERTIES)
    config['CLINVEC_PATH'] = str(CLINVEC)
    config['MAX_EPISODE_LEN_STEPS'] = MAX_EPISODE_LEN_STEPS
    config_path = tmp_path / 'dataset.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    assert extract_main([str(config_path)]) == 0
    return data_dir


@pytest.fixture(scope='module')
def widths():
    """94 / 37 / 16, derived from the live pair rather than declared."""
    config = yaml.safe_load(DATASET_CONFIG.read_text())
    props = yaml.safe_load(VARIABLE_PROPERTIES.read_text())
    counts = {'numeric': 0, 'categorical': 0, 'ordinal': 0}
    for feat in config['VALUED_FEATS']:
        counts[props[feat]['type']] += 1
    return counts


def load(data_dir, name):
    return np.load(data_dir / 'extracted' / f'{name}.npy')


def n_episodes(data_dir):
    return len(pd.read_csv(data_dir / 'labels.csv'))


def test_dense_arrays_match_the_shapes_and_dtypes_of_section_4_4(
    extracted, widths
):
    n, T = n_episodes(extracted), MAX_EPISODE_LEN_STEPS
    expected = {
        'val_times': ((n, T), np.int32),
        'val_masks': ((n, T), np.float32),
        'val_numeric_indicators': ((n, T, widths['numeric']), np.float32),
        'val_categorical_indicators':
            ((n, T, widths['categorical']), np.float32),
        'val_ordinal_indicators': ((n, T, widths['ordinal']), np.float32),
        'val_text_indicators': ((n, T, 1), np.float32),
        'val_drug_indicators': ((n, T, 1), np.float32),
        'event_indicators': ((n, T, 1), np.float32),
        'event_times': ((n, T), np.int32),
        'event_masks': ((n, T), np.float32),
        'index_times': ((n,), np.dtype('datetime64[ns]')),
        'time_to_event': ((n,), np.float32),
        'event_type': ((n,), np.int8),
    }
    for name, (shape, dtype) in expected.items():
        array = load(extracted, name)
        assert array.shape == shape, name
        assert array.dtype == dtype, name


def test_the_derived_widths_are_the_94_37_16_of_section_4_4(widths):
    """Section A.3 froze the contract at these counts. If this fails the
    config and the YAML have moved, and section 4.4 needs revisiting --
    it is not a licence to hard-code new numbers downstream."""
    assert widths == {'numeric': 94, 'categorical': 37, 'ordinal': 16}


def test_per_feature_value_arrays_match_section_4_4(extracted, widths):
    n, T = n_episodes(extracted), MAX_EPISODE_LEN_STEPS
    for i in range(widths['numeric']):
        array = load(extracted, f'val_numeric_values_{i}')
        assert array.shape == (n, T, 1) and array.dtype == np.float32
    for family, count in (('categorical', widths['categorical']),
                          ('ordinal', widths['ordinal'])):
        for i in range(count):
            array = load(extracted, f'val_{family}_values_{i}')
            assert array.shape == (n, T), f'{family}_{i}'
            assert array.dtype == np.int16, f'{family}_{i}'


def test_sparse_text_and_drug_arrays_match_section_4_4(extracted):
    n = n_episodes(extracted)
    text_offsets = load(extracted, 'text_offsets_0')
    text_values = load(extracted, 'text_values_0')
    text_timesteps = load(extracted, 'text_timesteps_0')
    assert text_offsets.shape == (n + 1,) and text_offsets.dtype == np.int64
    assert text_values.ndim == 1 and text_values.dtype == np.int32
    assert text_timesteps.shape == text_values.shape
    assert text_timesteps.dtype == np.int32
    assert text_offsets[-1] == len(text_values)

    drug_offsets = load(extracted, 'drug_offsets_0')
    drug_values = load(extracted, 'drug_values_0')
    assert drug_offsets.shape == (n + 1,) and drug_offsets.dtype == np.int64
    assert drug_values.shape[1] == 30 and drug_values.dtype == np.int32
    for name, dtype in (('drug_doses_0', np.float32),
                        ('drug_masks_0', np.float32),
                        ('drug_timesteps_0', np.int32)):
        array = load(extracted, name)
        assert array.dtype == dtype, name
        assert array.shape[0] == drug_values.shape[0], name


def test_no_static_data_and_no_per_episode_text_payload(extracted):
    """Section 4.4: no zero-width static array, and the per-episode text
    tokens and masks go away with the rest of the payload."""
    written = {p.name for p in (extracted / 'extracted').iterdir()}
    assert 'static_data.npy' not in written
    assert not [n for n in written if n.startswith('val_text_masks')]
    assert not [n for n in written if n.startswith('val_text_values')]


def test_metadata_and_ids_are_written(extracted):
    directory = extracted / 'extracted'
    for name in ('metadata.pkl', 'episode_ids.pkl', 'text_strings.pkl'):
        assert (directory / name).exists(), name
    with open(directory / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    assert metadata['lookup_feats'] == ['TEXT_SUPERFEATURE', 'DRUG']
    assert metadata['lookup_slot_dims'] == [1, 30]
    # The pad index is the ClinVec row count; the text table is C4's.
    assert metadata['lookup_pad_indices'][0] is None
    assert metadata['lookup_pad_indices'][1] > 0


def test_the_last_records_are_kept_at_the_right_of_the_row(extracted):
    """Sections 4.1 and 4.2 together, on real fixture episodes: the
    window ends at INDEX_TIME, keeps the most recent N, and is
    right-aligned so t = 0 is the final column."""
    times = load(extracted, 'val_times')
    masks = load(extracted, 'val_masks')
    index_times = load(extracted, 'index_times')
    with open(extracted / 'extracted' / 'episode_ids.pkl', 'rb') as f:
        ids = pickle.load(f)

    truncated = 0
    for row, (patid, stay_index) in enumerate(ids):
        stamps = pd.read_csv(
            extracted / 'root' / str(patid) / 'timeseries.csv',
            usecols=['TIMESTAMP']
        )['TIMESTAMP']
        stamps = pd.to_datetime(
            stamps, utc=True, format='ISO8601'
        ).sort_values()
        origin = pd.Timestamp(index_times[row]).tz_localize('UTC')
        window = stamps[stamps <= origin]
        if len(window) > MAX_EPISODE_LEN_STEPS:
            truncated += 1
        expected = window.iloc[-MAX_EPISODE_LEN_STEPS:]
        expected = ((expected - origin) //
                    pd.Timedelta(minutes=1)).to_numpy(np.int32)
        assert (times[row][masks[row] == 1] == expected).all(), row
        assert masks[row][-1] == 1.0 and times[row][-1] == 0
    assert truncated, "the window must actually bite for this to mean much"


def test_val_times_are_increasing_and_non_positive(extracted):
    """Invariant 7, over the fixture cohort."""
    times = load(extracted, 'val_times')
    masks = load(extracted, 'val_masks')
    assert (times <= 0).all()
    for row in range(times.shape[0]):
        observed = times[row][masks[row] == 1]
        assert (np.diff(observed) > 0).all(), row


def test_lookup_indices_are_in_range_for_their_tables(extracted):
    """Invariant 8, as far as C1 can check it: text indices against the
    string table it just assigned, drug indices against ClinVec."""
    with open(extracted / 'extracted' / 'text_strings.pkl', 'rb') as f:
        strings = pickle.load(f)
    text_values = load(extracted, 'text_values_0')
    assert ((text_values >= 0) & (text_values < len(strings))).all()
    assert all(s.strip() for s in strings), "no blank reaches the table"
    assert len(set(strings)) == len(strings), "deduped"

    with open(extracted / 'extracted' / 'metadata.pkl', 'rb') as f:
        pad = pickle.load(f)['lookup_pad_indices'][1]
    drug_values = load(extracted, 'drug_values_0')
    assert ((drug_values >= 0) & (drug_values <= pad)).all()


def test_csr_offsets_agree_with_the_dense_indicators(extracted):
    """The two representations of "this timestep has one" must not
    disagree, in either family."""
    for family in ('text', 'drug'):
        offsets = load(extracted, f'{family}_offsets_0')
        timesteps = load(extracted, f'{family}_timesteps_0')
        indicators = load(extracted, f'val_{family}_indicators')
        for row in range(indicators.shape[0]):
            in_csr = set(
                timesteps[offsets[row]:offsets[row + 1]].tolist()
            )
            dense = set(np.flatnonzero(indicators[row, :, 0] == 1).tolist())
            assert in_csr == dense, (family, row)


def test_targets_and_ids_follow_labels_csv(extracted):
    labels = pd.read_csv(extracted / 'labels.csv')
    with open(extracted / 'extracted' / 'episode_ids.pkl', 'rb') as f:
        ids = pickle.load(f)
    assert ids == list(zip(labels.PATID, labels.STAY_INDEX))
    assert np.allclose(load(extracted, 'time_to_event'),
                       labels.TIME_TO_EVENT.to_numpy(np.float32))
    assert (load(extracted, 'event_type') ==
            labels.EVENT_TYPE.to_numpy()).all()
