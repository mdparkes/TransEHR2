"""Probes for scoring each episode's most recent earlier hospital admission.

The load-bearing claim is that this reproduces the extraction's own rule for which diagnosis
records an episode can see. The extraction emits one record per row of `stays.csv`, timestamped
at that row's `DISCHTIME`, and blanks any whose timestamp falls after the episode's `INTIME`.
If the rule here admitted one more or one fewer admission, the index would be computed on a
different population than the cohort selects, and the two arms of the comparison would no
longer be paired -- so the boundary at `DISCHTIME == INTIME` and the exclusion of the episode's
own admission are both tested directly rather than assumed.

The second claim is that "most recent" means the latest discharge, not the previous row of the
file: `stays.csv` is ordered by ICU `INTIME`, and an admission that starts earlier can still
discharge later.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_charlson_logreg import (CODE_F, CODE_MISSING, GENDER_MAP, STATIC_FEATS,
                                 VARIABLE_PROPERTIES, _write_fold)

from compute_charlson_index import (build_cohort, build_table, check_folds_agree,
                                    collect_episodes, feature_availability,
                                    most_recent_earlier_admission, score_patient, write_cohort)
from TransEHR2.data.charlson import CONDITION_KEYS, WEIGHTS
from TransEHR2.data.cohorts import load_episode_manifest
from TransEHR2.data.statics import static_offsets

PATIENT_ID = 12345

# One code per admission, chosen so each scores a different, unmistakable index.
CODES = {
    # hadm: (icd_code, icd_version, comorbidity)
    100: ('410', 9, 'myocardial_infarction'),          # weight 1
    200: ('1970', 9, 'metastatic_solid_tumour'),       # weight 6
    300: ('3441', 9, 'hemiplegia_paraplegia'),         # weight 2
}


def _stays(rows):
    """A `stays.csv` frame prepared as `score_patient` prepares it.

    Args:
        rows: Per-admission (hadm_id, intime, dischtime) as ISO strings.

    Returns:
        The frame, datetimes parsed and sorted by INTIME.
    """
    frame = pd.DataFrame(rows, columns=['HADM_ID', 'INTIME', 'DISCHTIME'])
    for column in ('INTIME', 'DISCHTIME'):
        frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values('INTIME').reset_index(drop=True)


def _write_patient(tmp_path, stay_rows, codes=CODES, write_diagnoses=True):
    """Write a synthetic per-subject directory and return its path."""
    directory = tmp_path / str(PATIENT_ID)
    directory.mkdir()
    _stays(stay_rows).to_csv(directory / 'stays.csv', index=False)
    if write_diagnoses:
        pd.DataFrame(
            [{'HADM_ID': hadm, 'ICD_CODE': code, 'ICD_VERSION': version}
             for hadm, (code, version, _) in codes.items()],
        ).to_csv(directory / 'diagnoses.csv', index=False)
    return str(directory)


def _by_id(results):
    return {result['episode_id']: result for result in results}


# ---------------------------------------------------------------------------
# Which admission is scored
# ---------------------------------------------------------------------------

def test_the_first_stay_has_no_earlier_admission():
    stays = _stays([(100, '2130-01-01 12:00', '2130-01-10 09:00')])
    row, hours = most_recent_earlier_admission(stays, 1)
    assert row is None and hours is None


def test_the_latest_discharge_is_scored_not_the_previous_row():
    """Rows are ordered by ICU admission, so the previous row is not necessarily the most
    recent discharge: a long admission that started first can end last."""
    stays = _stays([
        (100, '2130-01-01 12:00', '2130-06-01 09:00'),  # starts first, discharges last
        (200, '2130-02-01 12:00', '2130-02-05 09:00'),
        (300, '2130-07-01 12:00', '2130-07-10 09:00'),  # the episode being scored
    ])
    row, hours = most_recent_earlier_admission(stays, 3)
    assert int(row['HADM_ID']) == 100
    assert hours == pytest.approx((pd.Timestamp('2130-07-01 12:00')
                                   - pd.Timestamp('2130-06-01 09:00')).total_seconds() / 3600)


def test_an_admission_discharged_exactly_at_admission_counts():
    """The extraction blanks a record whose timestamp is strictly after INTIME, so one
    timestamped at INTIME survives and is scorable."""
    stays = _stays([
        (100, '2130-01-01 12:00', '2130-01-10 09:00'),
        (200, '2130-01-10 09:00', '2130-01-20 09:00'),
    ])
    row, hours = most_recent_earlier_admission(stays, 2)
    assert int(row['HADM_ID']) == 100
    assert hours == pytest.approx(0.0)


def test_an_admission_discharged_after_admission_is_not_scored():
    """An overlapping admission's coding is not available at the time of the ICU admission,
    and the extraction blanks its record."""
    stays = _stays([
        (100, '2130-01-01 12:00', '2130-02-10 09:00'),  # discharges after the next INTIME
        (200, '2130-01-15 09:00', '2130-01-20 09:00'),
    ])
    row, hours = most_recent_earlier_admission(stays, 2)
    assert row is None and hours is None


def test_the_episodes_own_admission_is_never_scored():
    """A single-stay patient whose own discharge precedes its ICU admission would otherwise
    score itself, which would leak the current stay's coding into a feature."""
    stays = _stays([(100, '2130-01-10 12:00', '2130-01-05 09:00')])
    row, hours = most_recent_earlier_admission(stays, 1)
    assert row is None and hours is None


def test_an_admission_with_no_discharge_time_is_skipped():
    stays = _stays([
        (100, '2130-01-01 12:00', None),
        (200, '2130-02-01 12:00', '2130-02-05 09:00'),
        (300, '2130-03-01 12:00', '2130-03-05 09:00'),
    ])
    row, _ = most_recent_earlier_admission(stays, 3)
    assert int(row['HADM_ID']) == 200


# ---------------------------------------------------------------------------
# Scoring one patient
# ---------------------------------------------------------------------------

def test_each_episode_scores_its_own_most_recent_admission(tmp_path):
    directory = _write_patient(tmp_path, [
        (100, '2130-01-01 12:00', '2130-01-10 09:00'),
        (200, '2130-02-01 12:00', '2130-02-10 09:00'),
        (300, '2130-03-01 12:00', '2130-03-10 09:00'),
    ])
    results = _by_id(score_patient((directory, [(PATIENT_ID * 1000 + n, n)
                                                for n in (1, 2, 3)])))

    assert results[PATIENT_ID * 1000 + 1]['status'] == 'no_earlier_admission'

    second = results[PATIENT_ID * 1000 + 2]
    assert second['status'] == 'ok'
    assert second['source_hadm_id'] == 100
    assert second['charlson_index'] == WEIGHTS['myocardial_infarction']
    assert second['myocardial_infarction'] == 1
    assert second['n_conditions'] == 1

    third = results[PATIENT_ID * 1000 + 3]
    assert third['source_hadm_id'] == 200
    assert third['charlson_index'] == WEIGHTS['metastatic_solid_tumour']
    assert third['metastatic_solid_tumour'] == 1
    # The earlier admission's myocardial infarction is not carried forward: the index is that
    # of the most recent set of discharge diagnoses, not of the patient's whole history.
    assert third['myocardial_infarction'] == 0


def test_unmapped_codes_are_counted_rather_than_dropped_silently(tmp_path):
    """A cohort scoring zero everywhere would look like a healthy population; the count of
    codes that reached no comorbidity is what distinguishes that from a broken mapping."""
    codes = {100: ('4019', 9, None), 200: ('410', 9, 'myocardial_infarction')}
    directory = _write_patient(tmp_path, [
        (100, '2130-01-01 12:00', '2130-01-10 09:00'),
        (200, '2130-02-01 12:00', '2130-02-10 09:00'),
    ], codes=codes)
    result = _by_id(score_patient((directory, [(PATIENT_ID * 1000 + 2, 2)])))
    scored = result[PATIENT_ID * 1000 + 2]
    assert scored['charlson_index'] == 0
    assert scored['n_codes'] == 1
    assert scored['n_unmapped_codes'] == 1


def test_an_admission_with_no_codes_is_reported_not_scored_as_zero(tmp_path):
    """An index of 0 means no comorbidity was coded; an admission with no codes at all is a
    different thing and must not be reported as a healthy patient."""
    directory = _write_patient(tmp_path, [
        (100, '2130-01-01 12:00', '2130-01-10 09:00'),
        (200, '2130-02-01 12:00', '2130-02-10 09:00'),
    ], codes={200: ('410', 9, 'myocardial_infarction')})
    result = _by_id(score_patient((directory, [(PATIENT_ID * 1000 + 2, 2)])))
    assert result[PATIENT_ID * 1000 + 2]['status'] == 'no_codes'


@pytest.mark.parametrize('episode_number', [0, 4, -1])
def test_an_episode_outside_the_stays_file_is_reported(tmp_path, episode_number):
    directory = _write_patient(tmp_path, [
        (100, '2130-01-01 12:00', '2130-01-10 09:00'),
        (200, '2130-02-01 12:00', '2130-02-10 09:00'),
    ])
    result = _by_id(score_patient((directory, [(PATIENT_ID * 1000 + episode_number,
                                                episode_number)])))
    assert result[PATIENT_ID * 1000 + episode_number]['status'] == 'episode_out_of_range'


def test_missing_source_files_are_reported_per_episode(tmp_path):
    directory = _write_patient(tmp_path, [
        (100, '2130-01-01 12:00', '2130-01-10 09:00'),
        (200, '2130-02-01 12:00', '2130-02-10 09:00'),
    ], write_diagnoses=False)
    result = _by_id(score_patient((directory, [(PATIENT_ID * 1000 + 2, 2)])))
    assert result[PATIENT_ID * 1000 + 2]['status'] == 'missing_diagnoses'

    empty = tmp_path / 'nobody'
    empty.mkdir()
    result = _by_id(score_patient((str(empty), [(999001, 1)])))
    assert result[999001]['status'] == 'missing_stays'


# ---------------------------------------------------------------------------
# The output table
# ---------------------------------------------------------------------------

def test_the_table_holds_the_scored_episodes_and_counts_the_rest():
    results = [
        {'episode_id': 2002, 'status': 'ok', 'charlson_index': 3, 'n_conditions': 2,
         'source_hadm_id': 200, 'n_codes': 5, 'n_unmapped_codes': 1,
         'hours_before_admission': 48.0,
         **{key: 0 for key in CONDITION_KEYS}},
        {'episode_id': 1002, 'status': 'ok', 'charlson_index': 1, 'n_conditions': 1,
         'source_hadm_id': 100, 'n_codes': 2, 'n_unmapped_codes': 0,
         'hours_before_admission': 12.0,
         **{key: 0 for key in CONDITION_KEYS}},
        {'episode_id': 3001, 'status': 'no_earlier_admission'},
        {'episode_id': 4001, 'status': 'missing_stays'},
    ]
    frame, counts = build_table(results)

    assert counts == {'ok': 2, 'no_earlier_admission': 1, 'missing_stays': 1}
    assert len(frame) == 2
    # Sorted by episode id, with the identifiers split out for joining.
    assert list(frame['episode_id']) == [1002, 2002]
    assert list(frame['patient_id']) == [1, 2]
    assert list(frame['episode_number']) == [2, 2]
    assert 'status' not in frame.columns
    # Every comorbidity gets a column, in the module's declared order.
    assert list(frame.columns)[-len(CONDITION_KEYS):] == list(CONDITION_KEYS)


def test_an_empty_result_still_has_the_full_column_set():
    """The consumer joins on named columns, so a run that scored nothing must still write a
    readable header rather than an empty file."""
    frame, counts = build_table([{'episode_id': 1, 'status': 'missing_stays'}])
    assert frame.empty
    assert counts == {'missing_stays': 1}
    for column in ('episode_id', 'charlson_index', *CONDITION_KEYS):
        assert column in frame.columns


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def test_episodes_are_deduplicated_across_folds_and_partitions(tmp_path):
    """Folds partition the same episodes, so an episode is named in several listfiles. Scoring
    is a property of the episode, so it must be scored once."""
    for fold in ('fold1', 'fold2'):
        fold_dir = tmp_path / fold
        fold_dir.mkdir()
        for partition in ('train', 'val', 'test'):
            pd.DataFrame({
                'stay': [f'/data/{PATIENT_ID}/episode1_timeseries.csv'],
                'patient_id': [PATIENT_ID],
                'episode_number': [1],
            }).to_csv(fold_dir / f'{fold}_{partition}.csv', index=False)

    episodes = collect_episodes(str(tmp_path), ['fold1', 'fold2'])
    assert list(episodes) == [PATIENT_ID * 1000 + 1]
    patient_dir, episode_number = episodes[PATIENT_ID * 1000 + 1]
    assert os.path.basename(patient_dir) == str(PATIENT_ID)
    assert episode_number == 1


def test_no_listfiles_is_an_error_rather_than_an_empty_run(tmp_path):
    (tmp_path / 'fold1').mkdir()
    with pytest.raises(FileNotFoundError, match='listfiles'):
        collect_episodes(str(tmp_path), ['fold1'])


# ---------------------------------------------------------------------------
# The cohort manifest
# ---------------------------------------------------------------------------

def test_the_cohort_is_the_intersection_of_all_three_features():
    """Membership is availability of the features, not a proxy for it: an episode missing any
    one of the index, the age or the sex cannot be given to the regression at all."""
    scored = {1, 2, 3, 4, 5}
    extracted = {1, 2, 3, 4}          # 5 was never extracted
    has_age = {1, 2, 3}               # 4 has no age
    has_sex = {1, 2, 4}               # 3 has no sex

    cohort, funnel = build_cohort(scored, extracted, has_age, has_sex)
    assert cohort == [1, 2]
    assert [count for _, count in funnel] == [5, 4, 3, 2]


def test_a_scored_episode_outside_the_extraction_is_dropped():
    """It has an index but no arrays, so no model can be run on it."""
    cohort, _ = build_cohort({1, 2}, {1}, {1, 2}, {1, 2})
    assert cohort == [1]


def test_an_extracted_episode_with_no_index_is_dropped():
    cohort, _ = build_cohort({1}, {1, 2}, {1, 2}, {1, 2})
    assert cohort == [1]


def test_the_manifest_round_trips_through_the_loader_reader(tmp_path):
    """What this writes is what `cohorts.load_episode_manifest` reads, so the two must agree on
    the format -- the header comments included."""
    path = write_cohort(str(tmp_path / 'sub' / 'cohort.txt'), [1002, 2002, 3003],
                        'A test cohort.')
    assert list(load_episode_manifest(path)) == [1002, 2002, 3003]
    assert open(path).readline().startswith('# A test cohort.')


def test_the_availability_sets_come_from_the_extracted_arrays(tmp_path):
    """Age and sex are read at their cumulative-width offsets; a stored zero in either column
    means missing rather than a newborn or a category."""
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1001, True, 55.0, CODE_F, 0.0),    # both present
        (1002, True, 0.0, CODE_F, 1.0),     # no age
        (1003, True, 60.0, CODE_MISSING, 1.0),   # no sex
    ]})
    offsets = static_offsets(VARIABLE_PROPERTIES, STATIC_FEATS, 0)

    extracted, has_age, has_sex = feature_availability(
        data_dir, 'fold1', offsets, GENDER_MAP
    )
    assert extracted == {1001, 1002, 1003}
    assert has_age == {1001, 1003}
    assert has_sex == {1001, 1002}

    cohort, _ = build_cohort(extracted, extracted, has_age, has_sex)
    assert cohort == [1001]


def test_a_fold_short_of_a_cohort_episode_is_reported(tmp_path):
    """Folds are partitions of one episode set, so a fold missing a cohort episode would train
    the control on fewer episodes than the regression scores."""
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [
        (1001, True, 55.0, CODE_F, 0.0),
        (1002, True, 60.0, CODE_F, 1.0),
    ]})
    _write_fold(data_dir, 'fold2', {'train': [(1001, True, 55.0, CODE_F, 0.0)]})

    missing = check_folds_agree(data_dir, ['fold1', 'fold2'], [1001, 1002])
    assert missing['fold1'] == []
    assert missing['fold2'] == [1002]


def test_a_fold_with_no_extracted_partitions_is_not_reported_as_short(tmp_path):
    """An unextracted fold is a different problem from a fold that lost episodes, and calling
    it short would send the reader looking in the wrong place."""
    data_dir = str(tmp_path / 'data')
    _write_fold(data_dir, 'fold1', {'train': [(1001, True, 55.0, CODE_F, 0.0)]})
    os.makedirs(os.path.join(data_dir, 'fold2'))

    missing = check_folds_agree(data_dir, ['fold1', 'fold2'], [1001])
    assert missing == {'fold1': []}
