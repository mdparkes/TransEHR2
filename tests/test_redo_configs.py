"""Probes for the redo experiments' generated configs.

Two silent failures are possible here and neither shows in a result table.

The first is a hyperparameter that differs between experiments. Every column of a results table
is compared against that table's control, and the comparison only means what it claims if the
columns differ in which records reach the model and in nothing else. A config carrying a
different learning rate still trains, still reports, and still gets a P value.

The second is a switch set wrongly. The four switches are the experiment's entire definition --
its column heading is a claim about them -- and any combination of them produces a model that
trains. The redo experiments replace 10-17 one for one, so the pattern they have to reproduce
is already on disk, which makes it checkable rather than merely asserted.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_redo_configs import (CHARLSON_EXPERIMENT, EXPERIMENTS, build, load_base)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments', 'tuning', 'phase3_base.yaml')
EXPERIMENT_DIR = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments')

SWITCHES = ('USE_TEXT', 'USE_HISTORICAL_NONTEXT_RECORDS', 'USE_HISTORICAL_TEXT_RECORDS',
            'USE_INSTAY_RECORDS')

# The experiment each redo experiment replaces. Same design, new cutoff and new feature set.
REPLACES = {20: 10, 21: 11, 22: 12, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17}

# What has to be identical across every experiment for their columns to be comparable.
SHARED = ('POSITION_ENCODING', 'PRETRAIN_LEARNING_RATE', 'PRETRAIN_LR_HALF_LIFE',
          'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE', 'CMPNT_MASK_RATIO',
          'RECORD_MASK_RATIO', 'THP_PRED_LOSS_TIME_WT', 'OBS_UNOBS_SAMPLE_RATIO',
          'BATCH_SIZE', 'PRETRAIN_TOTAL_EPOCH', 'FINETUNE_TOTAL_EPOCH', 'HISTORY_LEN_STEPS',
          'EVENT_LADDER_P_MAX', 'VALUE_LADDER_P_MAX', 'PRETRAIN_SEED', 'FINETUNE_SEED')

HISTORY_LEN = 261
PRETRAIN_EPOCHS = 500


@pytest.fixture(scope='module')
def configs():
    """Every redo config, built from the real base."""
    base = load_base(BASE)
    return {entry[0]: build(base, entry, HISTORY_LEN, PRETRAIN_EPOCHS, 'data/charlson.txt')[1]
            for entry in EXPERIMENTS}


def old_config(number):
    """The config of the experiment a redo experiment replaces, or None if it is gone."""
    import glob
    matches = glob.glob(os.path.join(EXPERIMENT_DIR, f'experiment{number}_*.yaml'))
    if not matches:
        return None
    with open(matches[0]) as handle:
        return yaml.safe_load(handle)


# ------------------------------------------------------------------------------------------
# The switches
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('new,old', sorted(REPLACES.items()))
def test_each_experiment_reproduces_the_design_it_replaces(configs, new, old):
    """The redo changes the cutoff, the feature set and the cohort -- not the designs."""
    previous = old_config(old)
    if previous is None:
        pytest.skip(f'experiment{old} is no longer on disk to compare against')
    assert [bool(configs[new][key]) for key in SWITCHES] == \
           [bool(previous[key]) for key in SWITCHES], (
        f'experiment{new} does not have experiment{old}\'s switches'
    )


def test_no_two_experiments_share_a_design_within_a_cohort(configs):
    """Two identical columns in one table would be the same model reported twice."""
    seen = {}
    for number, config in configs.items():
        key = (config.get('COHORT_SUBSET'), tuple(bool(config[s]) for s in SWITCHES))
        assert key not in seen, f'experiment{number} duplicates experiment{seen[key]}'
        seen[key] = number


def test_a_model_reading_no_records_at_all_is_not_produced(configs):
    """Every switch off is an empty input, which trains and reports like anything else."""
    for number, config in configs.items():
        assert any(config[key] for key in SWITCHES[1:]), (
            f'experiment{number} reads neither historical nor peri-stay records'
        )


def test_text_features_are_built_wherever_text_records_are_read(configs):
    """USE_HISTORICAL_TEXT_RECORDS admits text into the history region, but USE_TEXT is what
    builds the features; on without the other is a silently text-free text arm."""
    for number, config in configs.items():
        if config['USE_HISTORICAL_TEXT_RECORDS']:
            assert config['USE_TEXT'], f'experiment{number} reads text records without text'


# ------------------------------------------------------------------------------------------
# The shared configuration
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('key', SHARED)
def test_every_experiment_carries_the_same_tuned_value(configs, key):
    values = {number: config.get(key) for number, config in configs.items()}
    assert len(set(map(repr, values.values()))) == 1, (
        f'{key} differs across the experiments, so their columns are not comparable: {values}'
    )


def test_the_tuned_values_are_the_ones_phase_2_settled(configs):
    """Read from the base rather than restated, so re-tuning moves both together."""
    base = load_base(BASE)
    for number, config in configs.items():
        for key in ('POSITION_ENCODING', 'PRETRAIN_LEARNING_RATE', 'PRETRAIN_LR_HALF_LIFE',
                    'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE', 'CMPNT_MASK_RATIO',
                    'RECORD_MASK_RATIO', 'THP_PRED_LOSS_TIME_WT'):
            assert config[key] == base[key], f'experiment{number} overrides {key}'


def test_the_selected_history_length_reaches_every_experiment(configs):
    """The base leaves it null so the sweep can vary it; an experiment that inherited the null
    would read the whole extracted history regardless of what the sweep chose."""
    assert load_base(BASE)['HISTORY_LEN_STEPS'] is None
    for number, config in configs.items():
        assert config['HISTORY_LEN_STEPS'] == HISTORY_LEN, f'experiment{number} was not set'


def test_pretraining_gets_the_final_budget_not_the_tuning_one(configs):
    """200 epochs ranked configurations; the reported models get the longer budget."""
    for number, config in configs.items():
        assert config['PRETRAIN_TOTAL_EPOCH'] == PRETRAIN_EPOCHS


# ------------------------------------------------------------------------------------------
# The cohorts
# ------------------------------------------------------------------------------------------

def test_the_cohorts_are_the_three_the_tables_report(configs):
    from TransEHR2.data.cohorts import COHORTS
    cohorts = {config.get('COHORT_SUBSET') for config in configs.values()}
    assert cohorts == {'any_text', 'any_history', None}
    for cohort in cohorts - {None}:
        assert cohort in COHORTS


def test_only_the_charlson_control_is_selected_by_an_episode_list(configs):
    """`COHORT_EPISODES` overrides the cohort predicate, so one left on another experiment
    would quietly shrink it to the Charlson population."""
    for number, config in configs.items():
        if number == CHARLSON_EXPERIMENT:
            assert config.get('COHORT_EPISODES')
            assert 'COHORT_SUBSET' not in config, (
                'the Charlson control names both a cohort and an episode list'
            )
        else:
            assert 'COHORT_EPISODES' not in config, f'experiment{number} names an episode list'


def test_the_charlson_control_matches_the_other_peri_stay_only_models(configs):
    """Table 1 compares it against a regression, and Tables 2 and the supplement compare the
    same design against history arms. It has to be the same model on a different population."""
    charlson = [bool(configs[CHARLSON_EXPERIMENT][key]) for key in SWITCHES]
    assert charlson == [bool(configs[20][key]) for key in SWITCHES]
    assert charlson == [bool(configs[25][key]) for key in SWITCHES]
