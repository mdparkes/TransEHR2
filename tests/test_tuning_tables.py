"""Probes for the hyperparameter tuning table writer.

The tables are read as a grid, so a cell landing under the wrong column is the failure that
matters and it is invisible in the output -- every number is plausible wherever it sits. The
pivot is therefore checked value by value.

The writer reads evaluation YAMLs rather than a manifest, which is what lets one tool serve a
factorial sweep, a one-at-a-time sweep and a config-list grid. `report_tuning_results.py`
needs a manifest and ranks one hyperparameter at a time, so it can report none of the first
and only part of the last.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_finetune_grid as grid
import report_tuning_tables as tables
from reporting.jmir.tables import strip_markup


def write_run(root, name, task, hyperparameters, block, metrics, fold='fold0'):
    """Write one run's evaluation YAML into a model tree."""
    stage = 'pretrained' if task == 'pretrain' else task
    directory = os.path.join(root, name, fold, stage, 'evaluation')
    os.makedirs(directory, exist_ok=True)
    filename = 'evaluation_pretrained.yaml' if task == 'pretrain' else f'evaluation_{task}.yaml'
    with open(os.path.join(directory, filename), 'w') as handle:
        yaml.safe_dump({'hyperparameters': hyperparameters, block: metrics}, handle)


@pytest.fixture
def tree(tmp_path):
    """Two arms x two rates x (one half-life and a flat schedule), with known AUPRCs."""
    root = str(tmp_path / 'models')
    for arm in ('additive', 'rope'):
        for rate in (1e-05, 5e-05):
            for half_life in (160, None):
                # Encode the coordinates in the value so a misplaced cell is detectable.
                value = (0.5 + 0.01 * (arm == 'rope') + 0.001 * (rate == 5e-05)
                         + 0.0001 * (half_life is None))
                write_run(root, f'phase_{arm}_{rate}_{half_life}', 'mortality',
                          {'POSITION_ENCODING': arm, 'FINETUNE_LEARNING_RATE': rate,
                           'FINETUNE_LR_HALF_LIFE': half_life},
                          'validation_scores', {'AUPRC': round(value, 4), 'AUROC': 0.8})
    return root


def cells_of(table):
    """Map (category, row label) to the row's cells, keyed on the stripped label."""
    out, group = {}, ''
    for row in table.rows:
        if row.kind == 'category':
            group = strip_markup(row.label)
        else:
            out[(group, strip_markup(row.label))] = row.cells
    return out


def rate_label(value):
    """The stripped form of a rate label, as cells_of keys on it."""
    return strip_markup(tables.format_rate(value))


def test_the_grid_puts_every_value_in_its_own_cell(tree):
    runs = tables.discover(tree, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    assert table.columns == ['160 Epochs', 'No decay']
    cells = cells_of(table)
    tpe, rope = (tables.ARM_HEADINGS['additive'], tables.ARM_HEADINGS['rope'])
    low, high = rate_label(1e-05), rate_label(5e-05)
    assert cells[(tpe, low)] == ['0.5000', '0.5001']
    assert cells[(tpe, high)] == ['0.5010', '0.5011']
    assert cells[(rope, low)] == ['0.5100', '0.5101']
    assert cells[(rope, high)] == ['0.5110', '0.5111']


def test_the_rate_axis_descends_and_the_half_life_axis_ascends():
    """The published tables lead with the largest rate and the shortest half-life."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'models')
        for rate in (6e-05, 0.0002, 0.0006):
            for half_life in (20, 60, 160):
                write_run(root, f'phase_{rate}_{half_life}', 'pretrain',
                          {'POSITION_ENCODING': 'additive',
                           'PRETRAIN_LEARNING_RATE': rate,
                           'PRETRAIN_LR_HALF_LIFE': half_life},
                          'val_losses', {'Optimization_Loss': 5.0})
        runs = tables.discover(root, ['phase_*'], 'fold0', 'pretrain')
        table = tables.build_grid(runs, 'PRETRAIN_LEARNING_RATE', 'PRETRAIN_LR_HALF_LIFE',
                                  'val:Optimization_Loss', 'S4', 'caption', 4)
    assert table.columns == ['20 Epochs', '60 Epochs', '160 Epochs']
    labels = [strip_markup(row.label) for row in table.rows if row.kind == 'metric']
    # strip_markup keeps a superscript visible as [-4] for the plain-text view.
    assert labels == [rate_label(0.0006), rate_label(0.0002), rate_label(6e-05)]


def test_a_flat_schedule_sorts_last_and_is_named(tree):
    runs = tables.discover(tree, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    # None is the limit of the decay axis, not a missing value, so it belongs at the end.
    assert table.columns[-1] == 'No decay'


def test_the_headings_use_the_house_wording(tree):
    runs = tables.discover(tree, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    assert table.stub_head == 'Learning Rate'
    labels = [strip_markup(row.label) for row in table.rows if row.kind == 'category']
    assert labels == ['Temporal Positional Encoding (TPE)', 'RoPE']


def test_repeats_in_one_cell_are_averaged_and_counted(tmp_path):
    """Seed repeats share a cell, so the cell has to say it is a mean of several runs."""
    root = str(tmp_path / 'models')
    for seed, value in ((0, 0.60), (1, 0.62)):
        write_run(root, f'phase_seed{seed}', 'mortality',
                  {'POSITION_ENCODING': 'additive', 'FINETUNE_LEARNING_RATE': 5e-05,
                   'FINETUNE_LR_HALF_LIFE': None},
                  'validation_scores', {'AUPRC': value})
    runs = tables.discover(root, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    assert cells_of(table)[(tables.ARM_HEADINGS['additive'],
                            rate_label(5e-05))] == ['0.6100 (n=2)']


def test_a_cell_with_no_run_is_marked_rather_than_blank(tmp_path):
    root = str(tmp_path / 'models')
    for rate, half_life in ((1e-05, 160), (5e-05, None)):
        write_run(root, f'phase_{rate}_{half_life}', 'mortality',
                  {'POSITION_ENCODING': 'additive', 'FINETUNE_LEARNING_RATE': rate,
                   'FINETUNE_LR_HALF_LIFE': half_life},
                  'validation_scores', {'AUPRC': 0.6})
    runs = tables.discover(root, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    cells = cells_of(table)
    assert tables.MISSING in cells[(tables.ARM_HEADINGS['additive'],
                                   rate_label(1e-05))]


def test_the_flat_layout_gives_one_row_per_run(tree):
    runs = tables.discover(tree, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_flat(runs, ['val:AUPRC'], 'S6', 'caption', 4)
    assert len([row for row in table.rows if row.kind == 'metric']) == len(runs)
    assert table.columns[-1] == 'Validation AUPRC'


def test_a_missing_metric_names_what_is_there(tree):
    runs = tables.discover(tree, ['phase_*'], 'fold0', 'mortality')
    with pytest.raises(SystemExit, match='val:AUPRC'):
        tables.check_metrics(runs, ['val:Nonsense'])


def test_no_matching_run_is_refused(tmp_path):
    """An empty pattern would otherwise render a table of nothing."""
    root = str(tmp_path / 'models')
    os.makedirs(root)
    with pytest.raises(SystemExit, match='No evaluation found'):
        tables.discover(root, ['nothing_*'], 'fold0', 'mortality')


def test_the_grid_excludes_the_control_runs(tmp_path):
    """A control sits at the reference cell's rate and schedule but replaces or freezes the
    encoder, so leaving it in averages it into the grid point that belongs there."""
    root = str(tmp_path / 'models')
    shared = {'POSITION_ENCODING': 'additive', 'FINETUNE_LEARNING_RATE': 5e-05,
              'FINETUNE_LR_HALF_LIFE': None}
    write_run(root, 'phase_lr5em05_hlflat', 'mortality', shared,
              'validation_scores', {'AUPRC': 0.6551})
    write_run(root, 'phase_ctl_random', 'mortality',
              dict(shared, FINETUNE_ENCODER_INIT='random'),
              'validation_scores', {'AUPRC': 0.5613})
    write_run(root, 'phase_ctl_frozen', 'mortality',
              dict(shared, FINETUNE_FREEZE_ENCODER=True),
              'validation_scores', {'AUPRC': 0.6083})

    runs = tables.discover(root, ['phase_*'], 'fold0', 'mortality')
    assert len(runs) == 3
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    # The grid run alone, and no (n=) annotation, so the cell matches 2a's shape.
    assert cells_of(table)[(tables.ARM_HEADINGS['additive'],
                            rate_label(5e-05))] == ['0.6551']


def test_the_flat_layout_keeps_the_control_runs(tmp_path):
    root = str(tmp_path / 'models')
    shared = {'POSITION_ENCODING': 'additive', 'FINETUNE_LEARNING_RATE': 5e-05}
    write_run(root, 'phase_lr5em05_hlflat', 'mortality', shared,
              'validation_scores', {'AUPRC': 0.6551})
    write_run(root, 'phase_ctl_random', 'mortality',
              dict(shared, FINETUNE_ENCODER_INIT='random'),
              'validation_scores', {'AUPRC': 0.5613})
    runs = tables.discover(root, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_flat(runs, ['val:AUPRC'], 'S6', 'caption', 4)
    assert len([row for row in table.rows if row.kind == 'metric']) == 2


# ------------------------------------------------------------------------------------------
# Hyperparameters a run did not record
# ------------------------------------------------------------------------------------------
#
# `RECORDED_HYPERPARAMETERS` in run_experiment.py is a fixed list. A sweep over a key outside
# it writes evaluations that carry no coordinate, so every run answers None on both axes and
# the whole sweep renders as one cell -- a table that looks finished and reports two numbers.
# The cell name still carries the coordinates, because the generator writes them there.

def write_unrecorded_run(root, arm, rate, half_life, auprc):
    """One cell whose evaluation names the arm only, as the finetuning grid's runs did."""
    name = f'phase2b_{arm}_lr{grid.token(rate)}_hl{grid.token(half_life)}'
    write_run(root, name, 'mortality', {'POSITION_ENCODING': arm},
              'validation_scores', {'AUPRC': auprc, 'AUROC': 0.8})
    return name


@pytest.fixture
def unrecorded_tree(tmp_path):
    """The finetuning grid's shape: two arms x four rates x (three half-lives + flat)."""
    root = str(tmp_path / 'models')
    for arm in ('additive', 'rope'):
        for i, rate in enumerate(UNRECORDED_RATES):
            for j, half_life in enumerate(UNRECORDED_HALF_LIVES):
                # Coordinates encoded in the value, so a misplaced cell is detectable.
                value = 0.5 + 0.1 * (arm == 'rope') + 0.01 * i + 0.001 * j
                write_unrecorded_run(root, arm, rate, half_life, round(value, 4))
    return root


UNRECORDED_RATES = (5e-05, 2.2e-05, 1e-05, 5e-06)
UNRECORDED_HALF_LIVES = (160.0, 60.0, 20.0, None)


def test_the_grid_resolves_a_sweep_the_runs_did_not_record(unrecorded_tree):
    runs = tables.discover(unrecorded_tree, ['phase2b_*'], 'fold0', 'mortality')
    assert len(runs) == 32
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    assert table.columns == ['20 Epochs', '60 Epochs', '160 Epochs', 'No decay']

    cells = cells_of(table)
    for arm in ('additive', 'rope'):
        for i, rate in enumerate(UNRECORDED_RATES):
            base = 0.5 + 0.1 * (arm == 'rope') + 0.01 * i
            # Column order is the half-life ascending, so the fixture's order 160/60/20/flat
            # reverses for the first three and the flat schedule stays last.
            assert cells[(tables.ARM_HEADINGS[arm], rate_label(rate))] == [
                f'{base + 0.002:.4f}', f'{base + 0.001:.4f}',
                f'{base:.4f}', f'{base + 0.003:.4f}',
            ]


def test_a_sweep_the_runs_did_not_record_still_drops_its_controls(tmp_path):
    """A control carries no rate token, so nothing recovers a coordinate for it either. It
    must be excluded rather than landing in the cell both axes read as None."""
    root = str(tmp_path / 'models')
    write_unrecorded_run(root, 'additive', 5e-05, None, 0.6551)
    write_run(root, 'phase2b_additive_random', 'mortality',
              {'POSITION_ENCODING': 'additive', 'FINETUNE_ENCODER_INIT': 'random'},
              'validation_scores', {'AUPRC': 0.5613})
    write_run(root, 'phase2b_additive_frozen', 'mortality',
              {'POSITION_ENCODING': 'additive', 'FINETUNE_FREEZE_ENCODER': True},
              'validation_scores', {'AUPRC': 0.6083})

    runs = tables.discover(root, ['phase2b_*'], 'fold0', 'mortality')
    assert len(runs) == 3
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    assert cells_of(table)[(tables.ARM_HEADINGS['additive'],
                            rate_label(5e-05))] == ['0.6551']


@pytest.mark.parametrize('value', [5e-05, 2.2e-05, 1e-05, 5e-06, 0.0006, 0.002,
                                   480.0, 160.0, 60.0, 20.0, 329.0, None])
def test_the_decoder_round_trips_the_token_the_generator_writes(value):
    """The two halves are in different modules, so a change to either would otherwise put
    every cell of the next grid in the wrong place without failing anything."""
    assert tables.decode_token(grid.token(value)) == value


def test_a_recorded_value_is_preferred_to_the_name(tmp_path):
    """The name is a fallback, not a second source: a run that records the key is the
    authority on what it ran, and a relinked or edited cell can disagree with its name."""
    data = {'hyperparameters': {'FINETUNE_LEARNING_RATE': 1e-05}}
    assert tables.hyperparameter(data, 'FINETUNE_LEARNING_RATE',
                                 'phase2b_additive_lr5em05_hl160') == 1e-05


def test_a_recorded_flat_schedule_is_not_read_as_a_missing_value(tmp_path):
    """None is the flat schedule and a recorded one has to stay flat, or a run that recorded
    no decay would be pulled into whatever cell its name names."""
    data = {'hyperparameters': {'FINETUNE_LR_HALF_LIFE': None}}
    assert tables.hyperparameter(data, 'FINETUNE_LR_HALF_LIFE',
                                 'phase2b_additive_lr5em05_hl160') is None


@pytest.mark.parametrize('name', ['phase2b_additive_random', 'phase2d_additive_seed3',
                                  'phase2b_additive_lrflat_hlflat'])
def test_a_name_carrying_no_rate_yields_no_rate(name):
    assert tables.from_name(name, 'FINETUNE_LEARNING_RATE') is None


def test_only_the_keys_the_generator_names_are_read_from_the_name():
    """Anything else in a name is not a value, so a mask ratio must not be invented from it."""
    assert tables.from_name('phase2c_additive_lr5em05_hl160', 'CMPNT_MASK_RATIO') is None


def test_the_finetuning_schedule_is_recorded_from_now_on():
    """The name fallback exists for the runs already on disk. Leaving the keys off the
    recorded list would keep every later grid dependent on a naming convention."""
    import run_experiment
    for key in ('FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE'):
        assert key in run_experiment.RECORDED_HYPERPARAMETERS, (
            f'{key} is swept but not recorded, so its runs carry no coordinate'
        )
