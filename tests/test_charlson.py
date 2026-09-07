"""Probes for the Charlson comorbidity index.

Three claims are load-bearing. The published code sets must resolve codes in the form MIMIC-IV
stores them -- dot-free, and in either vocabulary -- because a comorbidity that never matches
is silently absent rather than wrong. The hierarchy must cancel the mild member of each pair,
or a single organ system is counted twice and the index runs past its documented ceiling. And
the index must be a function of the *set* of comorbidities, not of how many codes support each
one, since a discharge summary lists the same condition under several codes routinely.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransEHR2.data.charlson import (CODE_SETS, CONDITION_KEYS, HIERARCHY, MAX_INDEX, WEIGHTS,
                                     charlson_conditions, charlson_index, conditions_for_code,
                                     normalize_code)


def _present(codes):
    """The comorbidity keys a code set establishes, as a set."""
    return {key for key, flag in charlson_conditions(codes).items() if flag}


# ---------------------------------------------------------------------------
# Code resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('code,version,expected', [
    # ICD-9-CM, in the dot-free form MIMIC-IV stores.
    ('410', 9, 'myocardial_infarction'),
    ('41041', 9, 'myocardial_infarction'),
    ('42831', 9, 'congestive_heart_failure'),
    ('39891', 9, 'congestive_heart_failure'),
    ('4254', 9, 'congestive_heart_failure'),
    ('V434', 9, 'peripheral_vascular_disease'),
    ('36234', 9, 'cerebrovascular_disease'),
    ('2941', 9, 'dementia'),
    ('4928', 9, 'chronic_pulmonary_disease'),
    ('7100', 9, 'rheumatic_disease'),
    ('53170', 9, 'peptic_ulcer_disease'),
    ('5715', 9, 'mild_liver_disease'),
    ('25000', 9, 'diabetes_uncomplicated'),
    ('25042', 9, 'diabetes_complicated'),
    ('3441', 9, 'hemiplegia_paraplegia'),
    ('5856', 9, 'renal_disease'),
    ('1749', 9, 'malignancy'),
    ('5722', 9, 'moderate_severe_liver_disease'),
    ('1970', 9, 'metastatic_solid_tumour'),
    ('042', 9, 'hiv_aids'),
    # ICD-10.
    ('I2110', 10, 'myocardial_infarction'),
    ('I252', 10, 'myocardial_infarction'),
    ('I5023', 10, 'congestive_heart_failure'),
    ('I7025', 10, 'peripheral_vascular_disease'),
    ('I639', 10, 'cerebrovascular_disease'),
    ('G3109', 10, None),
    ('G311', 10, 'dementia'),
    ('J449', 10, 'chronic_pulmonary_disease'),
    ('M0579', 10, 'rheumatic_disease'),
    ('K2570', 10, 'peptic_ulcer_disease'),
    ('K740', 10, 'mild_liver_disease'),
    ('E119', 10, 'diabetes_uncomplicated'),
    ('E1122', 10, 'diabetes_complicated'),
    ('G8220', 10, 'hemiplegia_paraplegia'),
    ('N186', 10, 'renal_disease'),
    ('C3491', 10, 'malignancy'),
    ('K7291', 10, 'moderate_severe_liver_disease'),
    ('C7801', 10, 'metastatic_solid_tumour'),
    ('B20', 10, 'hiv_aids'),
])
def test_codes_resolve_to_their_published_comorbidity(code, version, expected):
    resolved = conditions_for_code(code, version)
    if expected is None:
        assert resolved == (), f'{code} (ICD-{version}) should flag nothing, got {resolved}'
    else:
        assert expected in resolved, f'{code} (ICD-{version}) -> {resolved}'


@pytest.mark.parametrize('code,version', [
    ('1730', 9),      # malignant neoplasm of skin, excluded from the malignancy set
    ('4019', 9),      # essential hypertension, in no Charlson set
    ('I1', 10),       # too short to reach any prefix
    ('', 9),
    (None, 9),
    ('E119', 9),      # an ICD-10 code offered as ICD-9 must not match
    ('25000', 10),    # and the converse
])
def test_codes_outside_the_sets_flag_nothing(code, version):
    assert conditions_for_code(code, version) == ()


def test_an_unknown_vocabulary_flags_nothing():
    """MIMIC-IV carries versions 9 and 10; anything else is not a lookup this can serve."""
    for version in (0, 11, None, 'icd10'):
        assert conditions_for_code('I2110', version) == ()


@pytest.mark.parametrize('written,stored', [
    ('410.41', '41041'),
    ('  I25.2 ', 'I252'),
    ('i252', 'I252'),
    ('V43.4', 'V434'),
])
def test_dotted_and_untidy_codes_normalize_to_the_stored_form(written, stored):
    assert normalize_code(written) == stored
    assert conditions_for_code(written, 9 if stored[0].isdigit() else 10) == \
        conditions_for_code(stored, 9 if stored[0].isdigit() else 10)


def test_a_code_may_flag_two_comorbidities():
    """Published behaviour, not an oversight: 437.3 is in the peripheral vascular set and also
    inside the cerebrovascular range 430-438, and 404.03 is in both the heart failure and
    renal sets."""
    assert set(conditions_for_code('4373', 9)) == {'peripheral_vascular_disease',
                                                   'cerebrovascular_disease'}
    assert set(conditions_for_code('40403', 9)) == {'congestive_heart_failure',
                                                    'renal_disease'}


# ---------------------------------------------------------------------------
# Hierarchy and weighting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('severe,mild,severe_code,mild_code,version', [
    ('diabetes_complicated', 'diabetes_uncomplicated', '25042', '25000', 9),
    ('moderate_severe_liver_disease', 'mild_liver_disease', '5722', '5715', 9),
    ('metastatic_solid_tumour', 'malignancy', '1970', '1749', 9),
])
def test_the_severe_comorbidity_cancels_the_mild_one(severe, mild, severe_code, mild_code,
                                                     version):
    both = _present([(severe_code, version), (mild_code, version)])
    assert severe in both
    assert mild not in both
    # The weight is the severe one alone, not the sum of the pair.
    assert charlson_index([(severe_code, version), (mild_code, version)]) == WEIGHTS[severe]
    # And the mild one still counts on its own.
    assert _present([(mild_code, version)]) == {mild}


def test_the_index_sums_the_weights_of_the_comorbidities_present():
    # Myocardial infarction (1) + hemiplegia (2) + AIDS/HIV (6).
    codes = [('410', 9), ('3441', 9), ('042', 9)]
    assert _present(codes) == {'myocardial_infarction', 'hemiplegia_paraplegia', 'hiv_aids'}
    assert charlson_index(codes) == 1 + 2 + 6


def test_repeated_codes_for_one_comorbidity_are_counted_once():
    """The index is a function of the set of comorbidities. A discharge summary listing heart
    failure under several codes must not score it several times."""
    once = charlson_index([('42831', 9)])
    many = charlson_index([('42831', 9), ('42832', 9), ('4280', 9), ('I5023', 10)])
    assert once == many == WEIGHTS['congestive_heart_failure']


def test_no_codes_is_an_index_of_zero():
    assert charlson_index([]) == 0
    assert _present([]) == set()
    assert charlson_conditions([]) == {key: False for key in CONDITION_KEYS}


def test_the_ceiling_is_reached_by_every_comorbidity_at_once():
    """MAX_INDEX is documented as the top of the scale, so the scale must attain it: taking one
    code from every set in both vocabularies must score exactly that, with the mild members of
    the hierarchical pairs cancelled."""
    codes = [(prefix, version)
             for version, code_set in CODE_SETS.items()
             for prefixes in code_set.values()
             for prefix in prefixes]
    present = charlson_conditions(codes)
    assert all(present[key] for key in CONDITION_KEYS
               if key not in {mild for _, mild in HIERARCHY})
    assert charlson_index(codes) == MAX_INDEX == 29


def test_every_comorbidity_is_reachable_in_both_vocabularies():
    """A set that no code can match would make the comorbidity permanently absent, which looks
    exactly like a healthy population rather than like a broken mapping."""
    for version, code_set in CODE_SETS.items():
        assert set(code_set) == set(CONDITION_KEYS), f'ICD-{version} does not cover every key'
        for key, prefixes in code_set.items():
            assert prefixes, f'ICD-{version} {key} has no codes'
            for prefix in prefixes:
                assert key in conditions_for_code(prefix, version), \
                    f'ICD-{version} {prefix} does not resolve to {key}'
