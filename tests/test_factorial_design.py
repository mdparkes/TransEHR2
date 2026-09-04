"""Probes for the factorial sweep design.

A factorial is run where two hyperparameters interact, which is the case in which the best
pair is not the pair of individual bests. The design therefore has to rank whole cells, and
the machinery built for a coordinate sweep has to refuse it rather than quietly average over
it.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hp_tuning.reporting import build_all_tables
from hp_tuning.results import rank_cells
from hp_tuning.spec import expand_trials, load_spec


GRID = {
    'PRETRAIN_LEARNING_RATE': {'values': [0.0002, 0.0006, 0.00006], 'select_on': 'pretrain'},
    'PRETRAIN_LR_HALF_LIFE': {'values': [160, 60, 20], 'select_on': 'pretrain'},
}


def make_spec(design='factorial', grid=None, arms=('additive', 'rope')):
    return {
        'SPEC_NAME': 'phase2a',
        'DESIGN': design,
        'ARMS': {name: {'POSITION_ENCODING': name} for name in arms},
        'ALIASES': {'PRETRAIN_LEARNING_RATE': 'lr', 'PRETRAIN_LR_HALF_LIFE': 'hl'},
        'GRID': grid if grid is not None else GRID,
    }


def write_spec(tmp_path, extra=None):
    """A spec on disk, with a base config beside it, so load_spec can validate it."""
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump({'MODEL_DIR': str(tmp_path / 'models')}))
    spec = {
        'SPEC_NAME': 'phase2a',
        'BASE_CONFIG': 'base.yaml',
        'DATASET_CONFIG': 'base.yaml',
        'OUTPUT_DIR': str(tmp_path / 'out'),
        'MANIFEST': str(tmp_path / 'manifest.yaml'),
        'FOLD': 'fold0',
        'ARMS': {'additive': {'POSITION_ENCODING': 'additive'}},
        'GRID': GRID,
        'DESIGN': 'factorial',
    }
    spec.update(extra or {})
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec))
    return str(path)


# --- expansion ---------------------------------------------------------------------------

def test_a_factorial_runs_every_combination():
    trials = expand_trials(make_spec())
    assert len(trials) == 3 * 3 * 2


def test_an_additive_sweep_of_the_same_grid_is_smaller():
    additive = expand_trials(make_spec(design='additive'))
    factorial = expand_trials(make_spec())
    assert len(additive) == (1 + 2 + 2) * 2
    assert len(factorial) > len(additive)


def test_every_cell_is_a_distinct_assignment():
    trials = expand_trials(make_spec(arms=('additive',)))
    cells = [tuple(sorted(t['cell'].items())) for t in trials]
    assert len(set(cells)) == len(cells)


def test_exactly_one_cell_per_arm_is_the_centre():
    trials = expand_trials(make_spec())
    for arm in ('additive', 'rope'):
        centres = [t for t in trials if t['arm'] == arm and t['is_centre']]
        assert len(centres) == 1
        assert centres[0]['cell'] == {'PRETRAIN_LEARNING_RATE': 0.0002,
                                      'PRETRAIN_LR_HALF_LIFE': 160}


def test_a_cell_carries_both_coordinates_into_its_overrides():
    trials = expand_trials(make_spec(arms=('rope',)))
    trial = next(t for t in trials if t['cell'] == {'PRETRAIN_LEARNING_RATE': 0.0006,
                                                    'PRETRAIN_LR_HALF_LIFE': 20})
    assert trial['overrides']['PRETRAIN_LEARNING_RATE'] == 0.0006
    assert trial['overrides']['PRETRAIN_LR_HALF_LIFE'] == 20
    assert trial['overrides']['POSITION_ENCODING'] == 'rope'


def test_a_cell_name_names_both_coordinates():
    trials = expand_trials(make_spec(arms=('additive',)))
    names = {t['name'] for t in trials}
    assert 'phase2a_additive_lr_0p0006_hl_20' in names


def test_a_cell_covers_nothing():
    # covers is how a coordinate lookup finds "the trial that ran this value". Every value
    # appears in several cells, so answering that lookup at all would be wrong.
    trials = expand_trials(make_spec())
    assert all(t['covers'] == [] for t in trials)


def test_extra_trials_still_hang_off_a_factorial():
    spec = make_spec(arms=('additive',))
    spec['EXTRA_TRIALS'] = [{
        'name': 'ablation', 'arm': 'additive', 'select_on': 'pretrain',
        'overrides': {'USE_TEXT': False},
    }]
    trials = expand_trials(spec)
    extras = [t for t in trials if t['is_extra']]
    assert len(extras) == 1
    assert extras[0]['overrides']['USE_TEXT'] is False
    # Built from the defaults, not from a cell, so it stays a one-variable comparison.
    assert extras[0]['overrides']['PRETRAIN_LR_HALF_LIFE'] == 160


# --- validation --------------------------------------------------------------------------

def test_an_unknown_design_is_refused(tmp_path):
    with pytest.raises(ValueError, match='DESIGN'):
        load_spec(write_spec(tmp_path, {'DESIGN': 'orthogonal'}))


def test_a_spec_without_a_design_is_additive(tmp_path):
    path = write_spec(tmp_path)
    spec = yaml.safe_load(open(path))
    del spec['DESIGN']
    open(path, 'w').write(yaml.safe_dump(spec))
    assert load_spec(path)['DESIGN'] == 'additive'


def test_a_factorial_grid_must_share_one_criterion(tmp_path):
    grid = dict(GRID)
    grid['CMPNT_MASK_RATIO'] = {'values': [0.25, 0.5], 'select_on': 'mortality'}
    with pytest.raises(ValueError, match='share one'):
        load_spec(write_spec(tmp_path, {'GRID': grid}))


def test_a_mixed_criterion_grid_is_fine_when_additive(tmp_path):
    grid = dict(GRID)
    grid['CMPNT_MASK_RATIO'] = {'values': [0.25, 0.5], 'select_on': 'mortality'}
    load_spec(write_spec(tmp_path, {'GRID': grid, 'DESIGN': 'additive'}))


# --- ranking -----------------------------------------------------------------------------

def fake_manifest(values, design='factorial'):
    """A manifest whose trials carry a pre-set metric, so ranking can be tested alone.

    Args:
        values: metric value per cell name suffix, e.g. {'lr_0p0002_hl_160': 1.0}.
        design: The design to record.
    """
    trials = []
    for suffix, value in values.items():
        lr, half_life = suffix.removeprefix('lr_').split('_hl_')
        trials.append({
            'name': f'phase2a_additive_{suffix}',
            'arm': 'additive',
            'is_centre': suffix == 'lr_0p0002_hl_160',
            'is_extra': False,
            'cell': {'PRETRAIN_LEARNING_RATE': lr, 'PRETRAIN_LR_HALF_LIFE': half_life},
            'covers': [],
            'needs_finetune': False,
            'config': '',
            'hyperparameter': None,
            'value': None,
            'select_on': 'pretrain',
            '_metric': value,
        })
    return {
        'spec_name': 'phase2a', 'design': design, 'arms': ['additive'],
        'fold': 'fold0', 'model_dir': '/nowhere', 'grid': GRID, 'trials': trials,
    }


@pytest.fixture
def stub_read_result(monkeypatch):
    """Replace the on-disk read with the metric baked into each fake trial."""
    import hp_tuning.results as results

    class Stub:
        def __init__(self, trial):
            self.trial = trial
            self.name = trial['name']
            self.value = trial['_metric']
            self.is_usable = trial['_metric'] is not None
            self.status = 'ok' if self.is_usable else 'pending'
            self.detail = ''
            self.grid_value = None

    monkeypatch.setattr(results, 'read_result', lambda m, t, c: Stub(t))
    return results


def test_the_best_cell_wins_not_the_best_coordinates(stub_read_result):
    # The pair (0p0006, 20) beats every cell, while 0p0002 and 160 each win their own margin.
    # A coordinate sweep would select (0p0002, 160), a cell that is not the best here.
    manifest = fake_manifest({
        'lr_0p0002_hl_160': 1.00, 'lr_0p0002_hl_20': 1.20,
        'lr_0p0006_hl_160': 1.30, 'lr_0p0006_hl_20': 0.50,
    })
    ranking = rank_cells(manifest, 'additive')
    assert ranking['best'].grid_value['PRETRAIN_LEARNING_RATE'] == '0p0006'
    assert ranking['best'].grid_value['PRETRAIN_LR_HALF_LIFE'] == '20'


def test_direction_is_taken_from_the_criterion(stub_read_result):
    manifest = fake_manifest({'lr_0p0002_hl_160': 1.0, 'lr_0p0006_hl_20': 0.5})
    # pretrain selects on a loss, so lower wins.
    assert rank_cells(manifest, 'additive')['direction'] == 'min'
    assert rank_cells(manifest, 'additive')['best'].value == 0.5


def test_an_unfinished_cell_makes_the_ranking_incomplete(stub_read_result):
    manifest = fake_manifest({'lr_0p0002_hl_160': 1.0, 'lr_0p0006_hl_20': None})
    ranking = rank_cells(manifest, 'additive')
    assert not ranking['complete']
    assert ranking['best'].value == 1.0


def test_no_usable_cell_leaves_no_winner(stub_read_result):
    manifest = fake_manifest({'lr_0p0002_hl_160': None, 'lr_0p0006_hl_20': None})
    assert rank_cells(manifest, 'additive')['best'] is None


def test_ranking_cells_refuses_an_additive_manifest(stub_read_result):
    manifest = fake_manifest({'lr_0p0002_hl_160': 1.0}, design='additive')
    with pytest.raises(ValueError, match='factorial'):
        rank_cells(manifest, 'additive')


def test_the_table_builder_refuses_a_factorial(stub_read_result):
    with pytest.raises(NotImplementedError, match='factorial'):
        build_all_tables(fake_manifest({'lr_0p0002_hl_160': 1.0}))
