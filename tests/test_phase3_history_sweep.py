"""Phase 3: the pre-admission history sweep, and the ablation that rides along with it.

Phase 3 needs no new sweep machinery -- it is the same additive shape as Phase 2 with a grid of
one -- so what is worth testing is the part that *is* new: `EXTRA_TRIALS`, the one-off ablations
that hang off a sweep without being cells in it.

The property that matters is isolation. An ablation has no ordered set of values to rank, so it
must be reported and never selected on. If it could win a ranking, a single ablation with a
lucky seed would silently redefine the hyperparameter the sweep exists to choose. The test that
seeds the ablation with the best score in the sweep and asserts the grid winner is unchanged is
the one to keep if any of these are ever pruned.
"""

import os

import pytest
import yaml

from hp_tuning.reporting import build_all_tables, write_csv
from hp_tuning.results import rank_hyperparameter
from hp_tuning.spec import (expand_trials, extra_trials, finetune_trials, load_spec,
                            pretrain_trials, write_manifest, write_trial_configs)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3_SPEC = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments', 'tuning',
                           'phase3_spec.yaml')
PHASE2_BASE = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments', 'tuning',
                           'phase2_base.yaml')

# What the plan's history arms are, in order. The first is the extraction capacity and is the
# no-crop centre; the last removes pre-admission history while leaving the in-stay episode.
HISTORY_ARMS = [500, 225, 100, 40, 5, 0]


def _write_spec(root, **overrides):
    """Materialise the real Phase 3 spec against a stand-in for Phase 2's assembled base.

    The shipped spec points at `phase3_base.yaml`, which does not exist until Phase 2 finishes
    and `select_tuned_hyperparameters.py` writes it. Standing in for it here keeps the test
    exercising the spec that will actually run, rather than a copy that can drift from it.
    """
    base = yaml.safe_load(open(PHASE2_BASE))
    base.update({'EXPERIMENT_NAME': 'phase3_base', 'POSITION_ENCODING': 'rope'})
    base_path = os.path.join(root, 'phase3_base.yaml')
    yaml.dump(base, open(base_path, 'w'), sort_keys=False)

    spec = yaml.safe_load(open(PHASE3_SPEC))
    spec.update({
        'BASE_CONFIG': base_path,
        'DATASET_CONFIG': os.path.join(REPO, 'TransEHR2/configs/datasets/mimic4.yaml'),
        'OUTPUT_DIR': os.path.join(root, 'configs'),
        'MANIFEST': os.path.join(root, 'manifest.yaml'),
    })
    spec.update(overrides)
    spec_path = os.path.join(root, 'spec.yaml')
    yaml.dump(spec, open(spec_path, 'w'), sort_keys=False)
    return spec_path


@pytest.fixture(scope='module')
def sweep(tmp_path_factory):
    """The expanded Phase 3 sweep, with its configs and manifest on disk."""
    root = str(tmp_path_factory.mktemp('phase3'))
    spec_path = _write_spec(root)
    spec = load_spec(spec_path)
    trials = write_trial_configs(spec, expand_trials(spec))
    manifest_path = write_manifest(spec, trials)
    manifest = yaml.safe_load(open(manifest_path))
    manifest['model_dir'] = os.path.join(root, 'models')
    yaml.dump(manifest, open(manifest_path, 'w'), sort_keys=False)
    return {'root': root, 'spec': spec, 'manifest': manifest,
            'manifest_path': manifest_path}


# ------------------------------------------------------------------------------------------
# Shape
# ------------------------------------------------------------------------------------------

def test_sweep_is_six_history_arms_plus_one_ablation(sweep):
    manifest = sweep['manifest']
    assert len(manifest['trials']) == 7
    # Every trial is selected on mortality, so the two stages are the same width.
    assert len(pretrain_trials(manifest)) == 7
    assert len(finetune_trials(manifest)) == 7
    assert len(extra_trials(manifest)) == 1


def test_every_history_arm_from_the_plan_is_present(sweep):
    manifest = sweep['manifest']
    ran = {}
    for trial in manifest['trials']:
        if trial.get('is_extra'):
            continue
        config = yaml.safe_load(open(trial['config']))
        ran[config['HISTORY_LEN_STEPS']] = trial['name']
    assert sorted(ran, key=lambda v: -v) == HISTORY_ARMS, (
        f'history arms on disk are {sorted(ran)}, expected {HISTORY_ARMS}'
    )


def test_the_zero_arm_is_a_real_crop_not_an_unset_value(sweep):
    """0 must survive as 0. `None` means "use everything extracted", the opposite arm."""
    trial = next(t for t in sweep['manifest']['trials'] if t['name'].endswith('hist_0'))
    config = yaml.safe_load(open(trial['config']))
    assert config['HISTORY_LEN_STEPS'] == 0
    assert config['HISTORY_LEN_STEPS'] is not None


def test_the_centre_is_the_uncropped_condition(sweep):
    """The centre sits at the extraction capacity, so it crops nothing and anchors the curve."""
    dataset = yaml.safe_load(open(os.path.join(
        REPO, 'TransEHR2/configs/datasets/mimic4.yaml')))
    centre = next(t for t in sweep['manifest']['trials'] if t['is_centre'])
    config = yaml.safe_load(open(centre['config']))
    assert config['HISTORY_LEN_STEPS'] == dataset['MAX_HISTORY_LEN_STEPS'], (
        'the centre must be the no-crop arm; if MAX_HISTORY_LEN_STEPS changes, the grid in '
        'phase3_spec.yaml has to change with it'
    )


# ------------------------------------------------------------------------------------------
# The ablation
# ------------------------------------------------------------------------------------------

def test_ablation_differs_from_the_centre_in_exactly_one_setting(sweep):
    """A head-to-head is only a head-to-head if nothing else moved."""
    manifest = sweep['manifest']
    centre = next(t for t in manifest['trials'] if t['is_centre'])
    ablation = extra_trials(manifest)[0]

    centre_config = yaml.safe_load(open(centre['config']))
    ablation_config = yaml.safe_load(open(ablation['config']))
    differing = {
        key for key in set(centre_config) | set(ablation_config)
        if centre_config.get(key) != ablation_config.get(key)
    }
    assert differing == {'EXPERIMENT_NAME', 'EVENT_LADDER_P_MAX'}, (
        f'ablation moves {differing - {"EXPERIMENT_NAME"}}; it must move EVENT_LADDER_P_MAX '
        f'alone'
    )


def test_ablation_puts_the_event_stream_on_the_value_ladder(sweep):
    """Direction check.

    The differentiated event ladder is already the default -- EVENT_LADDER_P_MAX = 3000 against
    the value stream's 7.9e6, asserted band-for-band in TransEHR2/test_rope_encoding.py. So the
    missing half of the comparison is the *shared* ladder, and that is what the ablation must
    run. Getting this backwards would run the default twice and measure nothing.
    """
    manifest = sweep['manifest']
    centre = yaml.safe_load(open(
        next(t for t in manifest['trials'] if t['is_centre'])['config']))
    ablation = yaml.safe_load(open(extra_trials(manifest)[0]['config']))

    assert centre['EVENT_LADDER_P_MAX'] < centre['VALUE_LADDER_P_MAX'], (
        'the centre no longer runs a narrowed event ladder, so this ablation is pointed the '
        'wrong way -- re-read the note in phase3_spec.yaml'
    )
    assert ablation['EVENT_LADDER_P_MAX'] == ablation['VALUE_LADDER_P_MAX'], (
        'the ablation must put the event stream on the value stream ladder'
    )


def test_ablation_cannot_win_a_ranking_even_with_the_best_score(sweep):
    """The isolation property, stated as the failure it prevents.

    The ablation is seeded with an AUPRC far above every grid arm. Selection must still return
    the best *grid* value, because an ablation is not a cell in the grid and has no business
    redefining the hyperparameter the sweep exists to choose.
    """
    manifest = dict(sweep['manifest'])
    root = os.path.join(sweep['root'], 'isolation')
    manifest['model_dir'] = root

    scores = {t['name']: 0.40 for t in manifest['trials']}
    scores[next(t['name'] for t in manifest['trials'] if t['name'].endswith('hist_100'))] = 0.55
    scores[extra_trials(manifest)[0]['name']] = 0.99

    for name, score in scores.items():
        directory = os.path.join(root, name, manifest['fold'], 'mortality', 'evaluation')
        os.makedirs(directory, exist_ok=True)
        yaml.dump({'validation_scores': {'AUPRC': score}},
                  open(os.path.join(directory, 'evaluation_mortality.yaml'), 'w'))

    ranking = rank_hyperparameter(manifest, 'tuned', 'HISTORY_LEN_STEPS')
    assert ranking['best'].grid_value == 100, (
        f"selection returned {ranking['best'].grid_value}; the ablation leaked into the grid"
    )
    assert 0.99 not in [r.value for r in ranking['results']]
    assert ranking['complete']


def test_ablation_is_reported_even_though_it_is_not_ranked(sweep):
    """Excluded from selection is not the same as invisible."""
    manifest = dict(sweep['manifest'])
    manifest['model_dir'] = os.path.join(sweep['root'], 'isolation')
    captions = [table.caption for table in build_all_tables(manifest)]
    assert any('Ablation' in caption for caption in captions), captions

    csv_path = write_csv(manifest, os.path.join(sweep['root'], 'report.csv'))
    body = open(csv_path).read()
    assert '<ablation:sharedladder>' in body


# ------------------------------------------------------------------------------------------
# Spec validation
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('mutation, needle', [
    ({'arm': 'nosucharm'}, 'not in ARMS'),
    ({'select_on': 'nosuchcriterion'}, 'expected one of'),
    ({'overrides': {}}, 'overrides nothing'),
    ({'overrides': {'HISTORY_LEN_STEPS': 40}}, 'the grid also sweeps'),
])
def test_spec_rejects_a_malformed_ablation(tmp_path, mutation, needle):
    spec = yaml.safe_load(open(PHASE3_SPEC))
    spec['EXTRA_TRIALS'][0].update(mutation)
    spec_path = _write_spec(str(tmp_path), EXTRA_TRIALS=spec['EXTRA_TRIALS'])
    with pytest.raises(ValueError) as error:
        load_spec(spec_path)
    assert needle in str(error.value), str(error.value)


def test_missing_base_config_names_the_step_that_writes_it(tmp_path):
    """Phase 3's base config is Phase 2's output, so its absence is a schedule error."""
    spec_path = _write_spec(str(tmp_path))
    spec = yaml.safe_load(open(spec_path))
    spec['BASE_CONFIG'] = str(tmp_path / 'not_written_yet.yaml')
    yaml.dump(spec, open(spec_path, 'w'), sort_keys=False)
    with pytest.raises(ValueError) as error:
        load_spec(spec_path)
    assert 'select_tuned_hyperparameters.py' in str(error.value)
