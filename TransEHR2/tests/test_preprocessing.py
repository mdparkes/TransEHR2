"""One test per claim C1 makes (blueprint sections 4.1-4.4, 5).

Every test names the reading it defends, so a future change that breaks
one can tell whether it is breaking a decision or a bug.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from TransEHR2.data.preprocessing import (
    OUT_OF_DOMAIN, _bucket_valued_feats, check_feature_contract,
    filter_timeseries_records, standardize_feats
)

from extract_data import main as extract_main


def run(mini):
    """Run the extractor over a MiniRoot and return the exit code."""
    config_path = mini.finish()
    return extract_main([str(config_path)])


# --- the lookup family (section 4.3) ---------------------------------

def test_bucket_returns_four_buckets_with_text_and_drug_together():
    """Section 4.3: the buckets are numeric / categorical / ordinal /
    lookup, and lookup holds both members of the family."""
    props = {
        'N': {'type': 'numeric'}, 'C': {'type': 'categorical'},
        'O': {'type': 'ordinal'}, 'T': {'type': 'text'},
        'D': {'type': 'drug'},
    }
    numeric, categorical, ordinal, lookup = _bucket_valued_feats(
        ['N', 'C', 'O'], ['T', 'D'], props
    )
    assert (numeric, categorical, ordinal) == (['N'], ['C'], ['O'])
    assert lookup == ['T', 'D']


def test_single_slot_lookup_writes_no_dose_or_mask_array(one_patient):
    """Section 4.4 lists drug_doses and drug_masks and no text
    counterpart: a one-slot feature's weight is 1 by definition, so both
    arrays would be constant."""
    assert run(one_patient) == 0
    names = {p.name for p in one_patient.extracted.iterdir()}
    assert 'text_values_0.npy' in names
    assert 'text_doses_0.npy' not in names
    assert 'text_masks_0.npy' not in names
    assert {'drug_doses_0.npy', 'drug_masks_0.npy'} <= names


def test_text_values_are_rank_one_drug_values_are_slotted(one_patient):
    """Section 4.4: text_values is (n_nonempty,), drug_values is
    (n_nonempty, n_slots)."""
    assert run(one_patient) == 0
    assert one_patient.load('text_values_0').ndim == 1
    assert one_patient.load('drug_values_0').shape[1] == 3


# --- the -1 sentinel (section 4.3) -----------------------------------

def test_out_of_domain_value_is_minus_one_and_counted(mini, capsys):
    """Section 4.3: index -1 means 'observed, but not in category_map'.
    Under index storage a miss must not fall through to category 0, which
    is a *false* label -- 'L' for ADMITCAT, level '0' for BLDUA -- rather
    than an absent one. The fixture's own values all map, so the case
    needs a value planted here."""
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', ''],
            # 'Z' is in no category_map.
            ['2019-01-02T00:00:00Z', 1.0, 'Z', '0', 'Few', '', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    assert run(mini) == 0

    indicators = mini.load('val_categorical_indicators')
    values = mini.load('val_categorical_values_0')
    # Last column is the index time, i.e. the row carrying 'Z'.
    assert indicators[0, -1, 0] == 1.0, "observed"
    assert values[0, -1] == OUT_OF_DOMAIN, "but not category 0"
    # The row before it is a value that does map.
    assert values[0, -2] == 0

    report = capsys.readouterr().out
    assert 'Out-of-domain values' in report
    assert 'CAT' in report


def test_unobserved_is_indicator_zero_and_index_minus_one(one_patient):
    """Section 4.3's table: (0, -1) is 'not observed'. The arrays are
    allocated with np.full(..., -1); np.zeros is the bug."""
    assert run(one_patient) == 0
    indicators = one_patient.load('val_categorical_indicators')
    values = one_patient.load('val_categorical_values_0')
    # The first timestep of the episode has no CAT value.
    assert indicators[0, 0, 0] == 0.0
    assert values[0, 0] == OUT_OF_DOMAIN


def test_padding_stays_at_the_sentinel(one_patient):
    """A padded timestep is not category 0 either: an episode shorter
    than T leaves the left of the row untouched."""
    one_patient.config['MAX_EPISODE_LEN_STEPS'] = 8
    assert run(one_patient) == 0
    masks = one_patient.load('val_masks')
    values = one_patient.load('val_categorical_values_0')
    assert (masks[0, :4] == 0).all(), "four of eight columns are padding"
    assert (values[masks == 0] == OUT_OF_DOMAIN).all()


def test_missing_cell_does_not_map_onto_a_None_level(one_patient):
    """Regression. ``groupby.first()`` leaves None, not NaN, in an object
    column, and ``str(None)`` is 'None' -- which is UB's declared level 0
    (as it is UBAC's). A missing cell mapped onto it would be a false
    label carrying indicator 0."""
    assert run(one_patient) == 0
    values = one_patient.load('val_ordinal_values_1')
    indicators = one_patient.load('val_ordinal_indicators')
    blank = indicators[0, :, 1] == 0
    assert blank.any(), "the patient has a blank UB cell"
    assert (values[0][blank] == OUT_OF_DOMAIN).all()


def test_na_colliding_level_is_read_as_a_value(one_patient):
    """Regression, the other half. Stage B writes na_rep='', so pandas'
    default NA strings must be off: 'None' is a real UB observation and
    under the defaults it would be read as missing, making that level
    unobservable on every timestep it was actually measured."""
    assert run(one_patient) == 0
    values = one_patient.load('val_ordinal_values_1')
    indicators = one_patient.load('val_ordinal_indicators')
    # Timestep 1 of 4 holds the literal 'None'.
    assert indicators[0, 1, 1] == 1.0
    assert values[0, 1] == 0


def test_numeric_looking_string_levels_map(one_patient):
    """Section 5's dtype trap: ORD's levels are the strings '0',
    '1-24', '25-50', as BLDUA's are. Read as a numeric column, 0.0 is
    not the key '0' and every value would miss."""
    assert run(one_patient) == 0
    values = one_patient.load('val_ordinal_values_0')
    indicators = one_patient.load('val_ordinal_indicators')
    observed = indicators[0, :, 0] == 1
    assert observed.sum() == 3
    assert sorted(values[0][observed].tolist()) == [0, 1, 2]


# --- the window (section 4.1) ----------------------------------------

def test_filter_keeps_the_last_records_not_the_first():
    """Section 4.1: this is the opposite end from MIMIC, whose filter
    kept the first N of a window."""
    index = pd.to_datetime(
        [f'2019-01-0{d} 00:00:00' for d in range(1, 6)], utc=True
    )
    start, stop = filter_timeseries_records(index, index[-1], 2)
    assert (start, stop) == (3, 5)
    assert list(index[start:stop]) == list(index[-2:])


def test_filter_excludes_records_after_the_index_time():
    """An episode's input is every record with TIMESTAMP <= INDEX_TIME."""
    index = pd.to_datetime(
        [f'2019-01-0{d} 00:00:00' for d in range(1, 6)], utc=True
    )
    start, stop = filter_timeseries_records(index, index[2], 10)
    assert (start, stop) == (0, 3)


def test_filter_includes_a_record_exactly_at_the_index_time():
    """Invariant 4: every INDEX_TIME names a timeseries.csv row, and
    section 4.2 puts t = 0 on the last record."""
    index = pd.to_datetime(['2019-01-01', '2019-01-02'], utc=True)
    assert filter_timeseries_records(index, index[1], 10) == (0, 2)


def test_filter_is_empty_with_no_record_at_or_before_the_origin():
    index = pd.to_datetime(['2019-01-05'], utc=True)
    start, stop = filter_timeseries_records(
        index, pd.Timestamp('2019-01-01', tz='UTC'), 10
    )
    assert start == stop == 0


# --- time and alignment (section 4.2) --------------------------------

def test_times_are_int32_minutes_before_the_index_time(one_patient):
    """Section 4.2: integer minutes, <= 0, int32. Not hours or days --
    float32 is not exact in those units at this range."""
    assert run(one_patient) == 0
    times = one_patient.load('val_times')
    masks = one_patient.load('val_masks')
    assert times.dtype == np.int32
    assert (times <= 0).all()
    assert times[0, -1] == 0, "t = 0 is the index time"
    assert times[0, -2] == -1440, "one day earlier"


def test_series_is_right_aligned_and_zero_padded_on_the_left(one_patient):
    """Section 4.2: the prediction origin lands in the final column of
    every row, so padding is on the left."""
    one_patient.config['MAX_EPISODE_LEN_STEPS'] = 6
    assert run(one_patient) == 0
    masks = one_patient.load('val_masks')
    assert masks[0].tolist() == [0, 0, 1, 1, 1, 1]


def test_no_empty_bins_between_distant_records(mini):
    """Section 5: ``.resample('1h')`` materializes every hourly bin
    between a patient's first and last record -- ~175,200 of them over a
    20-year history. A groupby on the minute-floored timestamp yields one
    timestep per source row and nothing else."""
    mini.config['MAX_EPISODE_LEN_STEPS'] = 4
    mini.add_patient(
        1001,
        timeseries=[
            ['2000-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', ''],
            ['2020-01-01T00:00:00Z', 2.0, 'U', '0', 'Few', '', ''],
        ],
        stays=[('DAD', '2000-01-01T00:00:00Z', '2020-01-01T00:00:00Z')],
    )
    assert run(mini) == 0
    masks = mini.load('val_masks')
    assert masks[0].sum() == 2, "two records, twenty years apart"


# --- drugs (sections 2.7, 4.3, 4.4) ----------------------------------

def test_drug_slots_are_padded_with_the_vocabulary_row_count(one_patient):
    """Section 4.4: unused slots take V, the pad row of a (V+1, 128)
    table. V is the ClinVec row count, not the cohort's largest index --
    a vocabulary entry nobody was dispensed would otherwise leave the pad
    colliding with a real drug."""
    assert run(one_patient) == 0
    values = one_patient.load('drug_values_0')
    doses = one_patient.load('drug_doses_0')
    masks = one_patient.load('drug_masks_0')
    assert values.shape == (1, 3)
    assert values[0].tolist() == [2, 3, 4], "slots 0 and 1, then pad"
    assert doses[0].tolist() == [1.0, 0.5, 0.0]
    assert masks[0].tolist() == [1.0, 1.0, 0.0]


def test_drug_masks_agree_with_the_pad_index(one_patient):
    """Section 4.4 keeps drug_masks even though it is derivable from
    ``values == V``; the two must not disagree."""
    assert run(one_patient) == 0
    values = one_patient.load('drug_values_0')
    masks = one_patient.load('drug_masks_0')
    assert (masks == (values != 4)).all()


def test_a_dispensation_with_no_valued_feature_still_gets_a_timestep(
    one_patient
):
    """Section 2.7: such a row is carried by REG alone, and invariant 4
    requires every drugs.csv timestamp to name a timeseries.csv row."""
    assert run(one_patient) == 0
    indicators = one_patient.load('val_drug_indicators')
    masks = one_patient.load('val_masks')
    assert masks[0, 0] == 1.0, "the drug-only row is a timestep"
    assert indicators[0, 0, 0] == 1.0


def test_a_drug_timestamp_naming_no_timeseries_row_is_an_error(mini):
    """Invariant 4's dispensation clause. With no DRUG column
    (section 2.6) the timestamp is the only join between the two files,
    so a stray one is a broken join, not a droppable row."""
    mini.add_patient(
        1001,
        timeseries=[['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-01T00:00:00Z')],
        drugs=[('2019-06-01T00:00:00Z', 0, 1, 1.0)],
    )
    with pytest.raises(RuntimeError, match='produced no data'):
        run(mini)


def test_over_cap_drugs_are_ignored(mini):
    """Section 2.7: SLOT = -1 marks a drug the 30-slot cap dropped. The
    rows are kept for the webapp; the extractor ignores them."""
    mini.add_patient(
        1001,
        timeseries=[['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-01T00:00:00Z')],
        drugs=[('2019-01-01T00:00:00Z', 0, 1, 1.0),
               ('2019-01-01T00:00:00Z', -1, 2, 1.0)],
    )
    assert run(mini) == 0
    values = mini.load('drug_values_0')
    assert values[0].tolist() == [1, 4, 4], "one real slot, two pads"


# --- the cohort-wide contract (sections 3, 4.4) ----------------------

def test_row_order_and_count_are_labels_csv_s(mini):
    """Section 3: labels.csv's row order is canonical and the fold
    arrays are positions in it, so extraction can neither reorder nor
    drop a row."""
    mini.config['MAX_EPISODE_LEN_STEPS'] = 4
    mini.add_patient(
        1002,
        timeseries=[['2019-01-02T00:00:00Z', 1.0, 'L', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-02T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    mini.add_patient(
        1001,
        timeseries=[['2019-01-01T00:00:00Z', 2.0, 'U', '0', 'Few', '', ''],
                    ['2019-01-03T00:00:00Z', 3.0, 'U', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-01T00:00:00Z'),
               ('DAD', '2019-01-03T00:00:00Z', '2019-01-03T00:00:00Z')],
    )
    assert run(mini) == 0
    import pickle
    with open(mini.extracted / 'episode_ids.pkl', 'rb') as f:
        ids = pickle.load(f)
    labels = pd.read_csv(mini.data_dir / 'labels.csv')
    assert ids == list(zip(labels.PATID, labels.STAY_INDEX))
    assert mini.load('val_times').shape[0] == len(labels)


def test_targets_are_columns_of_labels_csv(one_patient):
    """B4: time_to_event.npy and event_type.npy are columns of
    labels.csv, not a second derivation."""
    assert run(one_patient) == 0
    labels = pd.read_csv(one_patient.data_dir / 'labels.csv')
    assert one_patient.load('time_to_event').tolist() == \
        labels.TIME_TO_EVENT.tolist()
    assert one_patient.load('event_type').tolist() == \
        labels.EVENT_TYPE.tolist()
    assert one_patient.load('event_type').dtype == np.int8


def test_index_times_come_from_stays_in_labels_order(one_patient):
    """Section 3 gives labels.csv no INDEX_TIME column, so the value is
    joined from stays.csv while the order stays labels.csv's."""
    assert run(one_patient) == 0
    index_times = one_patient.load('index_times')
    assert index_times.dtype == np.dtype('datetime64[ns]')
    assert index_times[0] == np.datetime64('2019-01-04T00:00:00')


def test_no_static_data_file_when_static_feats_is_empty(one_patient):
    """Section 4.4: STATIC_FEATS is empty under A.3, so the array would
    be (n, 0). Omit it rather than writing a zero-width file."""
    assert run(one_patient) == 0
    assert not (one_patient.extracted / 'static_data.npy').exists()


def test_text_strings_are_interned_once_in_canonical_order(mini):
    """C1 assigns the row indices into text_embeddings.npy, first
    appearance in canonical row order winning index 0. C4 embeds the
    list in that order, which is what makes the index valid."""
    mini.config['MAX_EPISODE_LEN_STEPS'] = 4
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', 'first', ''],
            ['2019-01-02T00:00:00Z', 1.0, 'L', '0', 'Few', 'second', ''],
            ['2019-01-03T00:00:00Z', 1.0, 'L', '0', 'Few', 'first', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-03T00:00:00Z')],
    )
    assert run(mini) == 0
    import pickle
    with open(mini.extracted / 'text_strings.pkl', 'rb') as f:
        strings = pickle.load(f)
    assert strings == ['first', 'second'], "deduped, first-seen order"
    assert mini.load('text_values_0').tolist() == [0, 1, 0]


def test_blank_text_gets_no_csr_entry_and_indicator_zero(one_patient):
    """Section 4.4: the empty string must not reach the table."""
    assert run(one_patient) == 0
    indicators = one_patient.load('val_text_indicators')
    offsets = one_patient.load('text_offsets_0')
    assert indicators[0, :, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert offsets.tolist() == [0, 2]


def test_event_indicator_follows_presence_not_truth(mini):
    """Section A.3: process_event_data never consults the declared type,
    so a literal 0 would mark the timestep as an admission. Section 2.6
    requires ADMIT_DAD to be empty on a non-admission row; this is the
    behaviour that requirement exists for."""
    mini.add_patient(
        1001,
        timeseries=[['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', 0],
                    ['2019-01-02T00:00:00Z', 1.0, 'L', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    assert run(mini) == 0
    assert mini.load('event_masks')[0].sum() == 1, "the 0 is an event"
    assert mini.load('event_indicators')[0, -1, 0] == 1.0


def test_a_short_episode_is_reported_not_dropped(mini, capsys):
    """Section 3's fold arrays are positions in labels.csv, so extraction
    cannot skip a row. MIN_EPISODE_LEN_STEPS becomes a counter."""
    mini.config['MIN_EPISODE_LEN_STEPS'] = 3
    mini.add_patient(
        1001,
        timeseries=[['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-01T00:00:00Z')],
    )
    assert run(mini) == 0
    assert mini.load('val_times').shape[0] == 1, "the row is written"
    assert 'fewer than 3 timestep' in capsys.readouterr().out


# --- standardization (section 5) -------------------------------------

def test_standardize_saves_statistics_without_scaling_the_values(
    tmp_path
):
    """Section 5: extraction runs once for the cohort while
    standardization is per fold, so a single array cannot carry one
    fold's scaling. __getitem__ applies it at load time (C2)."""
    from TransEHR2.data.custom_types import TensorDimensions
    dims = TensorDimensions(
        n_episodes=4, max_ts_len=2, n_numeric_feats=1,
        n_categorical_feats=0, n_ordinal_feats=0, n_lookup_feats=0,
        n_event_feats=0, numeric_feat_dims=[1], categorical_feat_dims=[],
        ordinal_feat_dims=[], lookup_slot_dims=[], lookup_table_dims=[],
        lookup_pad_indices=[], static_feat_dims=[], static_total_dim=0,
    )
    values = np.arange(8, dtype=np.float32).reshape(4, 2, 1)
    arrays = {
        'val_numeric_indicators': np.ones((4, 2, 1), dtype=np.float32),
        'val_numeric_values': [values.copy()],
    }
    path = tmp_path / 'summary_statistics_fold0.npz'
    standardize_feats(arrays, dims, np.array([0, 1]), str(path))

    assert (arrays['val_numeric_values'][0] == values).all(), "untouched"
    stats = np.load(path)
    assert stats['means'][0] == pytest.approx(1.5), "train rows only"


def test_fold_statistics_are_written_per_fold(one_patient):
    """C2 loads summary_statistics_fold{i}.npz."""
    fold_dir = one_patient.data_dir / 'fold0'
    fold_dir.mkdir(parents=True)
    np.save(fold_dir / 'fold0_train_rows.npy', np.array([0]))
    assert run(one_patient) == 0
    assert (one_patient.extracted /
            'summary_statistics_fold0.npz').exists()


# --- invariant 12 (section 6) ----------------------------------------

def test_contract_holds_for_a_well_formed_pair():
    config = {'VALUED_FEATS': ['A'], 'EVENT_FEATS': [], 'TEXT_FEATS': [],
              'DRUG_FEATS': [], 'STATIC_FEATS': []}
    props = {'A': {'type': 'categorical', 'size': 2,
                   'category_map': {0: 'x', 1: 'y'}}}
    assert check_feature_contract(config, props) == []


def test_contract_fails_on_a_config_name_with_no_entry():
    """How AGE, SCU and ADMIT_DAD were missed (section A.3)."""
    config = {'VALUED_FEATS': ['A', 'B'], 'EVENT_FEATS': [],
              'TEXT_FEATS': [], 'DRUG_FEATS': [], 'STATIC_FEATS': []}
    props = {'A': {'type': 'numeric', 'size': 1}}
    failures = check_feature_contract(config, props)
    assert any("no variable_properties entry" in f for f in failures)


def test_contract_fails_on_a_stale_entry():
    """How CACS_RIW would have been caught. Equality, not containment."""
    config = {'VALUED_FEATS': ['A'], 'EVENT_FEATS': [], 'TEXT_FEATS': [],
              'DRUG_FEATS': [], 'STATIC_FEATS': []}
    props = {'A': {'type': 'numeric', 'size': 1},
             'GONE': {'type': 'numeric', 'size': 1}}
    failures = check_feature_contract(config, props)
    assert any("no config feature list" in f for f in failures)


def test_contract_fails_when_size_and_category_map_disagree():
    """Section A.3: this used to raise mid-extraction, on real data,
    hours in."""
    config = {'VALUED_FEATS': ['A'], 'EVENT_FEATS': [], 'TEXT_FEATS': [],
              'DRUG_FEATS': [], 'STATIC_FEATS': []}
    props = {'A': {'type': 'categorical', 'size': 3,
                   'category_map': {0: 'x', 1: 'y'}}}
    failures = check_feature_contract(config, props)
    assert any("keys must run 0..size-1" in f for f in failures)


def test_contract_reports_a_non_integer_map_key_rather_than_raising():
    """Regression. ``sorted(int(k) for k in cat_map)`` used to raise
    straight out of the checker on a hand-edited map, so the caller's
    per-failure report -- the whole point of checking up front -- never
    ran and nothing named the offending feature."""
    config = {'VALUED_FEATS': ['A'], 'EVENT_FEATS': [], 'TEXT_FEATS': [],
              'DRUG_FEATS': [], 'STATIC_FEATS': []}
    props = {'A': {'type': 'categorical', 'size': 2,
                   'category_map': {'x': 'L', 'y': 'U'}}}
    failures = check_feature_contract(config, props)
    assert any("'A'" in f and 'are not integers' in f
               for f in failures), failures


def test_extraction_refuses_to_start_on_a_broken_contract(one_patient):
    one_patient.var_properties['ORPHAN'] = {'type': 'numeric', 'size': 1}
    assert run(one_patient) == 1
    assert not one_patient.extracted.exists()


# --- re-running into a used directory (sections 4.3, 5.1) ------------

def test_a_rerun_clears_artifacts_of_a_removed_feature(one_patient):
    """Regression. Section 5.1's regression gate re-runs with
    ``DRUG_FEATS: []``; the previous run's drug arrays used to survive
    beside a metadata.pkl that no longer mentions them, and section 4.3
    has the webapp finding drugs by filename."""
    assert run(one_patient) == 0
    drug_files = {p.name for p in one_patient.extracted.iterdir()
                  if p.name.startswith(('drug_', 'val_drug_'))}
    assert drug_files, "the first run writes drug artifacts"

    # Same DATA_DIR, drug feature removed.
    one_patient.config['DRUG_FEATS'] = []
    del one_patient.var_properties['DRG']
    assert run(one_patient) == 0
    left = {p.name for p in one_patient.extracted.iterdir()
            if p.name.startswith(('drug_', 'val_drug_'))}
    assert not left, f"stale drug artifacts survived the re-run: {left}"


def test_a_rerun_keeps_the_directory_consistent_with_metadata(one_patient):
    """Every .npy left on disk must be one this run wrote."""
    one_patient.config['MAX_EPISODE_LEN_STEPS'] = 4
    assert run(one_patient) == 0
    (one_patient.extracted / 'val_numeric_values_99.npy').write_bytes(b'x')
    assert run(one_patient) == 0
    assert not (one_patient.extracted /
                'val_numeric_values_99.npy').exists()


# --- fold rows must address this cohort (section 3) ------------------

def test_fold_rows_past_the_end_of_the_cohort_are_refused(one_patient):
    """Section 3's fold arrays are positions in labels.csv. A stale fold
    directory used to surface as an IndexError from standardize_feats
    *after* the whole output directory had been written."""
    fold_dir = one_patient.data_dir / 'fold0'
    fold_dir.mkdir(parents=True)
    np.save(fold_dir / 'fold0_train_rows.npy', np.array([0, 7]))
    with pytest.raises(ValueError, match='indexes rows'):
        run(one_patient)
    assert not one_patient.extracted.exists(), "refused before writing"


def test_truncated_runs_skip_fold_statistics(one_patient, capsys):
    """``--n_examples`` yields a prefix of labels.csv, which the fold row
    indices do not address; statistics over it would mean nothing. The
    run used to write the whole directory and then die with an
    IndexError."""
    fold_dir = one_patient.data_dir / 'fold0'
    fold_dir.mkdir(parents=True)
    np.save(fold_dir / 'fold0_train_rows.npy', np.array([0]))
    config_path = one_patient.finish()
    assert extract_main([str(config_path), '-n', '1']) == 0
    assert not list(one_patient.extracted.glob('summary_statistics_*'))
    assert '--n_examples' in capsys.readouterr().out
