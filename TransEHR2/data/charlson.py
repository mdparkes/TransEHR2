"""The Charlson comorbidity index over a set of ICD diagnosis codes.

Comorbidities are identified with the enhanced ICD-9-CM and ICD-10 code sets of Quan et al.
(2005), and weighted with the original integer weights of Charlson et al. (1987). The two
vocabularies are kept separate because a code string alone is ambiguous: `250` is diabetes in
ICD-9-CM and nothing in ICD-10, and `I21` is a myocardial infarction in ICD-10 and not a
well-formed ICD-9-CM code. Every lookup therefore takes the version alongside the code.

Codes are matched by prefix on the dot-free form, which is how MIMIC-IV stores them: `4019`
rather than `401.9`, `I2510` rather than `I25.10`. A prefix is the correct matcher because the
published sets are stated over code families -- `428.x` names every subdivision of 428 -- and
because the two vocabularies differ in how deeply they subdivide.

One code may flag more than one comorbidity, and that is a property of the published algorithm
rather than an oversight here: `437.3` appears in the peripheral vascular set and also falls in
the cerebrovascular range `430`-`438`, and `404.03` appears in both the heart failure and renal
sets. Both are flagged.

Three pairs of comorbidities are hierarchical, and the severe member cancels the mild one so
that a single organ system is not counted twice:

    diabetes with chronic complication   cancels   diabetes without chronic complication
    moderate or severe liver disease     cancels   mild liver disease
    metastatic solid tumour              cancels   malignancy

The index is the sum of the weights of the comorbidities that survive that step, so it runs
from 0 to 29.

References:
    Charlson ME, Pompei P, Ales KL, MacKenzie CR. A new method of classifying prognostic
    comorbidity in longitudinal studies: development and validation. J Chronic Dis.
    1987;40(5):373-83.
    Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for defining comorbidities in
    ICD-9-CM and ICD-10 administrative data. Med Care. 2005;43(11):1130-9.
"""

from typing import Dict, Iterable, Sequence, Tuple

# ---------------------------------------------------------------------------
# Comorbidities and their weights
# ---------------------------------------------------------------------------

# (key, label, weight), in the order Quan et al. tabulate them. The order is the column order
# of any per-comorbidity table written from this module, so it is fixed here rather than left
# to dictionary insertion elsewhere.
CONDITIONS: Tuple[Tuple[str, str, int], ...] = (
    ('myocardial_infarction', 'Myocardial infarction', 1),
    ('congestive_heart_failure', 'Congestive heart failure', 1),
    ('peripheral_vascular_disease', 'Peripheral vascular disease', 1),
    ('cerebrovascular_disease', 'Cerebrovascular disease', 1),
    ('dementia', 'Dementia', 1),
    ('chronic_pulmonary_disease', 'Chronic pulmonary disease', 1),
    ('rheumatic_disease', 'Rheumatic disease', 1),
    ('peptic_ulcer_disease', 'Peptic ulcer disease', 1),
    ('mild_liver_disease', 'Mild liver disease', 1),
    ('diabetes_uncomplicated', 'Diabetes without chronic complication', 1),
    ('diabetes_complicated', 'Diabetes with chronic complication', 2),
    ('hemiplegia_paraplegia', 'Hemiplegia or paraplegia', 2),
    ('renal_disease', 'Renal disease', 2),
    ('malignancy', 'Malignancy', 2),
    ('moderate_severe_liver_disease', 'Moderate or severe liver disease', 3),
    ('metastatic_solid_tumour', 'Metastatic solid tumour', 6),
    ('hiv_aids', 'AIDS/HIV', 6),
)

CONDITION_KEYS: Tuple[str, ...] = tuple(key for key, _, _ in CONDITIONS)
CONDITION_LABELS: Dict[str, str] = {key: label for key, label, _ in CONDITIONS}
WEIGHTS: Dict[str, int] = {key: weight for key, _, weight in CONDITIONS}

# (severe, mild): presence of the severe comorbidity cancels the mild one.
HIERARCHY: Tuple[Tuple[str, str], ...] = (
    ('diabetes_complicated', 'diabetes_uncomplicated'),
    ('moderate_severe_liver_disease', 'mild_liver_disease'),
    ('metastatic_solid_tumour', 'malignancy'),
)

# The largest attainable index: every comorbidity present, with the hierarchy applied.
MAX_INDEX = sum(WEIGHTS[key] for key in CONDITION_KEYS
                if key not in {mild for _, mild in HIERARCHY})


# ---------------------------------------------------------------------------
# Code set construction
# ---------------------------------------------------------------------------

def _numeric_range(first: int, last: int, width: int = 3) -> Tuple[str, ...]:
    """Zero-padded numeric prefixes from `first` to `last` inclusive.

    ICD-9-CM chapter ranges are stated over the integer part of the code, and the integer part
    is zero-padded to three digits in the dot-free form -- `042` for HIV, not `42`. Expanding
    the range into one prefix per value keeps the matcher a pure prefix test rather than a
    numeric comparison that would have to re-parse every code.

    Args:
        first: First value in the range.
        last: Last value in the range, inclusive.
        width: Digits to pad to.

    Returns:
        One prefix string per value in the range.
    """
    return tuple(f'{value:0{width}d}' for value in range(first, last + 1))


def _subdivisions(stem: str, first: int, last: int) -> Tuple[str, ...]:
    """Prefixes `stem`+`first` .. `stem`+`last`, for a range within one code family.

    `425.4`-`425.9` and `E10.2`-`E10.5` are both of this shape: a fixed stem and a contiguous
    run of one further digit.

    Args:
        stem: Dot-free code stem, e.g. `'425'` or `'E10'`.
        first: First trailing digit.
        last: Last trailing digit, inclusive.

    Returns:
        One prefix string per digit in the range.
    """
    return tuple(f'{stem}{digit}' for digit in range(first, last + 1))


def _letter_range(letter: str, first: int, last: int) -> Tuple[str, ...]:
    """ICD-10 prefixes `letter``first` .. `letter``last`, two digits, inclusive.

    ICD-10 chapter ranges are stated as a letter and a two-digit number, e.g. `C00`-`C26`.

    Args:
        letter: Chapter letter.
        first: First two-digit number.
        last: Last two-digit number, inclusive.

    Returns:
        One prefix string per number in the range.
    """
    return tuple(f'{letter}{number:02d}' for number in range(first, last + 1))


# Enhanced ICD-9-CM code sets, Quan et al. (2005) Table 1. Dot-free prefixes.
ICD9_CODES: Dict[str, Tuple[str, ...]] = {
    'myocardial_infarction': ('410', '412'),
    'congestive_heart_failure': (
        ('39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491',
         '40493', '428')
        + _subdivisions('425', 4, 9)
    ),
    'peripheral_vascular_disease': (
        ('0930', '4373', '440', '441', '4471', '5571', '5579', 'V434')
        + _subdivisions('443', 1, 9)
    ),
    'cerebrovascular_disease': ('36234',) + _numeric_range(430, 438),
    'dementia': ('290', '2941', '3312'),
    'chronic_pulmonary_disease': (
        ('4168', '4169', '5064', '5081', '5088') + _numeric_range(490, 505)
    ),
    'rheumatic_disease': (
        ('4465', '7148', '725') + _subdivisions('710', 0, 4) + _subdivisions('714', 0, 2)
    ),
    'peptic_ulcer_disease': _numeric_range(531, 534),
    'mild_liver_disease': (
        ('07022', '07023', '07032', '07033', '07044', '07054', '0706', '0709', '570', '571',
         '5733', '5734', '5738', '5739', 'V427')
    ),
    'diabetes_uncomplicated': _subdivisions('250', 0, 3) + ('2508', '2509'),
    'diabetes_complicated': _subdivisions('250', 4, 7),
    'hemiplegia_paraplegia': (
        ('3341', '342', '343', '3449') + _subdivisions('344', 0, 6)
    ),
    'renal_disease': (
        ('40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493',
         '582', '585', '586', '5880', 'V420', 'V451', 'V56')
        + _subdivisions('583', 0, 7)
    ),
    'malignancy': (
        _numeric_range(140, 172) + _numeric_range(174, 194) + _subdivisions('195', 0, 8)
        + _numeric_range(200, 208) + ('2386',)
    ),
    'moderate_severe_liver_disease': (
        _subdivisions('456', 0, 2) + _subdivisions('572', 2, 8)
    ),
    'metastatic_solid_tumour': _numeric_range(196, 199),
    'hiv_aids': _numeric_range(42, 44),
}

# Enhanced ICD-10 code sets, Quan et al. (2005) Table 1. Dot-free prefixes.
ICD10_CODES: Dict[str, Tuple[str, ...]] = {
    'myocardial_infarction': ('I21', 'I22', 'I252'),
    'congestive_heart_failure': (
        ('I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I43', 'I50', 'P290')
        + _subdivisions('I42', 5, 9)
    ),
    'peripheral_vascular_disease': (
        'I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559',
        'Z958', 'Z959',
    ),
    'cerebrovascular_disease': (
        ('G45', 'G46', 'H340') + _letter_range('I', 60, 69)
    ),
    'dementia': ('F00', 'F01', 'F02', 'F03', 'F051', 'G30', 'G311'),
    'chronic_pulmonary_disease': (
        ('I278', 'I279', 'J684', 'J701', 'J703')
        + _letter_range('J', 40, 47) + _letter_range('J', 60, 67)
    ),
    'rheumatic_disease': (
        'M05', 'M06', 'M315', 'M32', 'M33', 'M34', 'M351', 'M353', 'M360',
    ),
    'peptic_ulcer_disease': _letter_range('K', 25, 28),
    'mild_liver_disease': (
        ('B18', 'K709', 'K717', 'K73', 'K74', 'K760', 'K768', 'K769', 'Z944')
        + _subdivisions('K70', 0, 3) + _subdivisions('K71', 3, 5)
        + _subdivisions('K76', 2, 4)
    ),
    'diabetes_uncomplicated': tuple(
        f'E{family}{digit}'
        for family in (10, 11, 12, 13, 14)
        for digit in (0, 1, 6, 8, 9)
    ),
    'diabetes_complicated': tuple(
        f'E{family}{digit}'
        for family in (10, 11, 12, 13, 14)
        for digit in (2, 3, 4, 5, 7)
    ),
    'hemiplegia_paraplegia': (
        ('G041', 'G114', 'G801', 'G802', 'G81', 'G82', 'G839')
        + _subdivisions('G83', 0, 4)
    ),
    'renal_disease': (
        ('I120', 'I131', 'N18', 'N19', 'N250', 'Z940', 'Z992')
        + _subdivisions('N03', 2, 7) + _subdivisions('N05', 2, 7)
        + _subdivisions('Z49', 0, 2)
    ),
    'malignancy': (
        _letter_range('C', 0, 26) + _letter_range('C', 30, 34) + _letter_range('C', 37, 41)
        + ('C43',) + _letter_range('C', 45, 58) + _letter_range('C', 60, 76)
        + _letter_range('C', 81, 85) + ('C88',) + _letter_range('C', 90, 97)
    ),
    'moderate_severe_liver_disease': (
        'I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767',
    ),
    'metastatic_solid_tumour': _letter_range('C', 77, 80),
    'hiv_aids': ('B20', 'B21', 'B22', 'B24'),
}

CODE_SETS: Dict[int, Dict[str, Tuple[str, ...]]] = {9: ICD9_CODES, 10: ICD10_CODES}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def normalize_code(code) -> str:
    """Reduce a code to the dot-free, upper-case form the code sets are stated in.

    MIMIC-IV already stores codes without dots, but a code that has come via a display string
    or a hand-written list may carry one, and trailing whitespace survives some CSV writers.

    Args:
        code: An ICD code, as a string or anything with a string form.

    Returns:
        The normalized code, or `''` if there is nothing left after stripping.
    """
    if code is None:
        return ''
    return str(code).strip().upper().replace('.', '').replace(' ', '')


def _build_prefix_index(code_set: Dict[str, Tuple[str, ...]]):
    """Group one vocabulary's prefixes by length, longest first.

    Matching walks the candidate prefixes of a code from longest to shortest and collects every
    comorbidity that matches at any length, so the prefixes are bucketed by length once here
    rather than scanned linearly per code.

    Args:
        code_set: Mapping from comorbidity key to its prefixes.

    Returns:
        A list of `(length, {prefix: (key, ...)})` pairs, in decreasing length.
    """
    by_length: Dict[int, Dict[str, Tuple[str, ...]]] = {}
    for key, prefixes in code_set.items():
        for prefix in prefixes:
            bucket = by_length.setdefault(len(prefix), {})
            bucket[prefix] = bucket.get(prefix, ()) + (key,)
    return [(length, by_length[length]) for length in sorted(by_length, reverse=True)]


_PREFIX_INDEX = {version: _build_prefix_index(code_set)
                 for version, code_set in CODE_SETS.items()}


def conditions_for_code(code, version) -> Tuple[str, ...]:
    """The comorbidities one ICD code flags.

    Args:
        code: An ICD code in any of the forms `normalize_code` accepts.
        version: ICD version, 9 or 10.

    Returns:
        Comorbidity keys, in `CONDITION_KEYS` order. Empty if the code flags none, if it is
        blank, or if the version is not one the code sets cover.
    """
    try:
        index = _PREFIX_INDEX[int(version)]
    except (KeyError, TypeError, ValueError):
        return ()

    normalized = normalize_code(code)
    if not normalized:
        return ()

    matched = set()
    for length, bucket in index:
        if length > len(normalized):
            continue
        matched.update(bucket.get(normalized[:length], ()))
    return tuple(key for key in CONDITION_KEYS if key in matched)


def charlson_conditions(codes: Iterable[Sequence]) -> Dict[str, bool]:
    """Which comorbidities a set of coded diagnoses establishes, after the hierarchy.

    Args:
        codes: Iterable of `(code, version)` pairs. Order and repetition do not matter; the
            index is a function of the set of comorbidities, not of how many codes support
            each one.

    Returns:
        A dict over every key in `CONDITION_KEYS`, `True` where the comorbidity is present.
        The mild member of a hierarchical pair is `False` when the severe member is present,
        so summing `WEIGHTS` over the `True` entries gives the index directly.
    """
    present = {key: False for key in CONDITION_KEYS}
    for entry in codes:
        code, version = entry[0], entry[1]
        for key in conditions_for_code(code, version):
            present[key] = True

    for severe, mild in HIERARCHY:
        if present[severe]:
            present[mild] = False
    return present


def charlson_index(codes: Iterable[Sequence]) -> int:
    """The Charlson comorbidity index of a set of coded diagnoses.

    Args:
        codes: Iterable of `(code, version)` pairs.

    Returns:
        The weighted sum, between 0 and `MAX_INDEX`.
    """
    present = charlson_conditions(codes)
    return sum(WEIGHTS[key] for key, flagged in present.items() if flagged)
