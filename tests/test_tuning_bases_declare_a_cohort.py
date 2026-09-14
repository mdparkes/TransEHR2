"""Every tuning base must name the population it tunes on.

Omitting `COHORT_SUBSET` is not an error and does not fail a run: the dataset simply loads
every episode. So a base that forgets it tunes against the whole extraction, more than half of
which carries no pre-admission record for any arm to read, and nothing in the log says so. The
selected hyperparameters would then be the ones that suit a population no reported comparison
is made on.

`run_experiment` validates the value against `COHORTS` but never requires the key, which is
what leaves the gap these probes close.
"""

import glob
import os

import pytest
import yaml

from TransEHR2.data.cohorts import COHORTS


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.join(REPO_ROOT, 'TransEHR2', 'configs', 'experiments')

# The hand-maintained bases a sweep is generated from, plus the sequence-length sweep. The
# generated per-trial configs are ignored rather than tracked, and inherit whatever the base
# they came from declares.
BASES = sorted(glob.glob(os.path.join(TUNING, 'tuning', 'phase*_base.yaml')))
BASES.append(os.path.join(TUNING, 'tune_sequence_lengths.yaml'))


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def test_there_are_bases_to_check():
    """A glob that silently matched nothing would make every probe below vacuous."""
    assert len(BASES) >= 8, f'found only {len(BASES)} tuning bases'


@pytest.mark.parametrize('path', BASES, ids=[os.path.basename(p) for p in BASES])
def test_the_base_declares_its_cohort(path):
    config = _load(path)
    assert 'COHORT_SUBSET' in config, (
        f'{os.path.basename(path)} names no COHORT_SUBSET, so it would tune on every episode '
        f'in the extraction rather than on a cohort'
    )


@pytest.mark.parametrize('path', BASES, ids=[os.path.basename(p) for p in BASES])
def test_the_declared_cohort_is_a_real_one(path):
    """`run_experiment` refuses an unknown name, but only once a trial has been queued."""
    cohort = _load(path).get('COHORT_SUBSET')
    assert cohort in COHORTS, f'{os.path.basename(path)} names cohort {cohort!r}'


@pytest.mark.parametrize('path', BASES, ids=[os.path.basename(p) for p in BASES])
def test_the_bases_agree_on_the_cohort(path):
    """Trials from different bases are compared with each other -- the encoding arms head to
    head, and the sequence-length sweep against the phase it inherits from -- so a base tuning
    on a different population would make those comparisons confound the two."""
    reference = _load(BASES[0])['COHORT_SUBSET']
    assert _load(path)['COHORT_SUBSET'] == reference


@pytest.mark.parametrize('path', BASES, ids=[os.path.basename(p) for p in BASES])
def test_no_base_names_a_retired_record_switch(path):
    """USE_HISTORICAL_RECORDS was split into the nontext and text switches. Unknown keys are
    not validated, so the old name sits in a config being silently ignored."""
    assert 'USE_HISTORICAL_RECORDS' not in _load(path), (
        f'{os.path.basename(path)} names USE_HISTORICAL_RECORDS, which nothing reads'
    )
