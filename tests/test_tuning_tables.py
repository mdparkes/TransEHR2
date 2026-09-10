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
    """Map (category, row label) to the row's cells."""
    out, group = {}, ''
    for row in table.rows:
        if row.kind == 'category':
            group = strip_markup(row.label)
        else:
            out[(group, strip_markup(row.label))] = row.cells
    return out


def test_the_grid_puts_every_value_in_its_own_cell(tree):
    runs = tables.discover(tree, ['phase_*'], 'fold0', 'mortality')
    table = tables.build_grid(runs, 'FINETUNE_LEARNING_RATE', 'FINETUNE_LR_HALF_LIFE',
                              'val:AUPRC', 'S5', 'caption', 4)
    assert table.columns == ['160 Epochs', 'No decay']
    cells = cells_of(table)
    tpe, rope = (tables.ARM_HEADINGS['additive'], tables.ARM_HEADINGS['rope'])
    assert cells[(tpe, '1e-05')] == ['0.5000', '0.5001']
    assert cells[(tpe, '5e-05')] == ['0.5010', '0.5011']
    assert cells[(rope, '1e-05')] == ['0.5100', '0.5101']
    assert cells[(rope, '5e-05')] == ['0.5110', '0.5111']


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
    assert table.stub_head == 'Finetuning Learn Rate'
    labels = [strip_markup(row.label) for row in table.rows if row.kind == 'category']
    assert labels == ['Temporal Positional Encoding (TPE)',
                      'Rotary Position Embedding (RoPE)']


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
    assert cells_of(table)[(tables.ARM_HEADINGS['additive'], '5e-05')] == ['0.6100 (n=2)']


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
    assert tables.MISSING in cells[(tables.ARM_HEADINGS['additive'], '1e-05')]


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
    assert cells_of(table)[(tables.ARM_HEADINGS['additive'], '5e-05')] == ['0.6551']


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
