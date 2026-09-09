"""Probes for the diagnosis label prevalence table.

The table is read as a description of the extracted population, so the two things that have to
hold are that each column's denominator is its own cohort and that a label's row is the same
label in every column. Both are easy to get wrong by one index, and neither shows up as an
error -- only as a wrong number in a supplementary table.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import report_diagnosis_prevalence as prevalence
from reporting.jmir.tables import strip_markup


NAMES = ['Shock', 'Acute and unspecified renal failure', 'Heart failure']


def make_data(labels, cohorts):
    """Assemble the dict `collect` returns from a label matrix and cohort masks."""
    return {
        'labels': np.asarray(labels, dtype=np.float32),
        'patients': np.arange(len(labels)),
        'cohorts': {key: np.asarray(mask, dtype=bool) for key, mask in cohorts.items()},
    }


def cells_by_label(table):
    """Map each row's label to its list of cells."""
    return {strip_markup(row.label): row.cells for row in table.rows if row.kind == 'metric'}


def test_rows_are_alphabetized():
    data = make_data([[1, 1, 0], [0, 1, 1]],
                     {'all': [True, True], 'history': [True, True], 'text': [True, False]})
    table = prevalence.build_table(data, NAMES, 'S3', 'Diagnosis labels')
    labels = [strip_markup(row.label) for row in table.rows if row.kind == 'metric']
    assert labels == sorted(NAMES)


def test_prevalence_is_relative_to_each_column_cohort():
    # Episode 0 carries Shock and is outside the text cohort; episode 1 carries renal failure
    # and heart failure and is inside it.
    data = make_data([[1, 0, 0], [0, 1, 1]],
                     {'all': [True, True], 'history': [True, True], 'text': [False, True]})
    table = prevalence.build_table(data, NAMES, 'S3', 'Diagnosis labels')
    cells = cells_by_label(table)

    # all: 1 of 2. text: 0 of 1, not 0 of 2.
    assert cells['Shock'] == ['1 (0.50)', '1 (0.50)', '0 (0.00)']
    assert cells['Heart failure'] == ['1 (0.50)', '1 (0.50)', '1 (1.00)']


def test_column_headings_carry_the_cohort_size():
    data = make_data([[1, 0, 0], [0, 1, 1], [0, 0, 1]],
                     {'all': [True] * 3, 'history': [True, True, False],
                      'text': [False, True, False]})
    table = prevalence.build_table(data, NAMES, 'S3', 'Diagnosis labels')
    assert table.columns[0].endswith('(n=3)')
    assert table.columns[1].endswith('(n=2)')
    assert table.columns[2].endswith('(n=1)')


def test_a_label_keeps_its_column_across_cohorts():
    """A label present only in one cohort must not shift into a neighbouring row."""
    # Only the middle label is ever positive, and only for the episode in the text cohort.
    data = make_data([[0, 0, 0], [0, 1, 0]],
                     {'all': [True, True], 'history': [True, True], 'text': [False, True]})
    table = prevalence.build_table(data, NAMES, 'S3', 'Diagnosis labels')
    cells = cells_by_label(table)
    assert cells['Acute and unspecified renal failure'] == ['1 (0.50)', '1 (0.50)', '1 (1.00)']
    assert cells['Shock'] == ['0 (0.00)', '0 (0.00)', '0 (0.00)']
    assert cells['Heart failure'] == ['0 (0.00)', '0 (0.00)', '0 (0.00)']


def test_label_count_mismatch_is_refused():
    data = make_data([[1, 0], [0, 1]],
                     {'all': [True, True], 'history': [True, True], 'text': [True, True]})
    with pytest.raises(SystemExit, match='different extractions'):
        prevalence.build_table(data, NAMES, 'S3', 'Diagnosis labels')


def test_empty_cohort_is_refused():
    """An empty column would divide by zero, so it stops the run rather than printing nan."""
    data = make_data([[1, 0, 0], [0, 1, 0]],
                     {'all': [True, True], 'history': [True, True], 'text': [False, False]})
    with pytest.raises(SystemExit, match='empty'):
        prevalence.build_table(data, NAMES, 'S3', 'Diagnosis labels')
