"""
Integration tests for the Phase 2 single-GPU infrastructure, on CPU against synthetic data.

`test_phase2_pipeline.py` is the real gate -- it runs on a GPU node against MIMIC-IV and is
what says whether the sweep can be launched. It cannot run here, and the orchestration it
exercises is a good deal of code to ship untested, so these tests do the same walk against a
five-episode synthetic fold small enough to run on a laptop CPU in under a minute:

    generate_tuning_configs.py -> run_experiment.py (pretrain) -> run_experiment.py (finetune)
        -> report_tuning_results.py -> select_tuned_hyperparameters.py

What that catches is everything structural: a config key that never reaches the model, an
evaluation YAML written where the reporter does not look, encoder weights the finetune stage
cannot find, a manifest index that does not line up with the SLURM array. What it cannot catch
is anything about memory, throughput, or the data -- which is exactly the division of labour
between this file and the GPU test.

The extraction layout is synthesised by `tests/test_seq_len_crop._build_arrays`, which already
produces arrays in the shape `extract_mimic()` writes.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import test_seq_len_crop as fixture
from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.preprocessing import save_dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Working directory for every subprocess, set by the `sweep` fixture. Keeps the relative
# ./log and ./checkpoints trees the runner writes out of the repository.
_SCRATCH = [None]

# One episode per row: (n_val_history, n_val_episode, n_event_history, n_event_episode).
# Six episodes rather than five so that a batch of two divides every partition evenly and the
# mortality labels the fixture assigns (i % 2) are balanced, which the AUPRC computation needs.
EPISODE_SPECS = [
    (0, 6, 0, 6),
    (2, 6, 5, 3),
    (5, 5, 1, 6),
    (4, 6, 3, 6),
    (7, 4, 7, 4),
    (3, 6, 2, 6),
]
MAX_HISTORY_LEN_STEPS = 8
MAX_EPISODE_LEN_STEPS = 6

VALUED_FEATS = ['numeric_a', 'numeric_b', 'categorical_a', 'ordinal_a', 'multilabel_a']
EVENT_FEATS = ['event_a', 'event_b', 'event_c']
TEXT_FEATS = ['note']
STATIC_FEATS = ['static_a', 'static_b']


def _write_variable_properties(path):
    """Write a variable_properties.yaml matching the fixture's array widths.

    Args:
        path: Destination YAML path.
    """
    # 'size' is the number of value COLUMNS the feature occupies in the concatenated encoder
    # input, and for a categorical or ordinal feature it must equal len(category_map): those
    # columns are the classes, one per column, which is how data/variable_properties.yaml is
    # written (GCS motor response: size 6, six levels). Declaring them independently produces a
    # shape error deep inside the discriminator, several frames from anything naming a config
    # key, so the widths are taken from the fixture's own constants here.
    properties = {
        'numeric_a': {'type': 'numeric', 'size': fixture.NUMERIC_DIMS[0]},
        'numeric_b': {'type': 'numeric', 'size': fixture.NUMERIC_DIMS[1]},
        'categorical_a': {
            'type': 'categorical', 'size': fixture.CATEGORICAL_DIMS[0],
            'category_map': {f'c{i}': i for i in range(fixture.CATEGORICAL_DIMS[0])},
        },
        'ordinal_a': {
            'type': 'ordinal', 'size': fixture.ORDINAL_DIMS[0],
            'category_map': {f'o{i}': i for i in range(fixture.ORDINAL_DIMS[0])},
        },
        'multilabel_a': {'type': 'multilabel', 'size': fixture.MULTILABEL_DIMS[0]},
    }
    for name in EVENT_FEATS + TEXT_FEATS + STATIC_FEATS:
        properties[name] = {'type': 'numeric', 'size': 1}
    with open(path, 'w') as f_out:
        yaml.dump(properties, f_out)


def _write_fold(fold_dir):
    """Write train, val and test partitions of the synthetic fold.

    All three partitions carry the same episodes. That would be meaningless for a measurement
    and is fine for a plumbing test, which only asks whether each partition is found, loaded
    and scored.

    Args:
        fold_dir: Directory to write the partitions into.
    """
    for partition in ('train', 'val', 'test'):
        arrays = fixture._build_arrays(
            EPISODE_SPECS, MAX_HISTORY_LEN_STEPS, MAX_EPISODE_LEN_STEPS
        )
        # The fixture writes multilabel values modulo 3, i.e. 0, 1 and 2. Multilabel targets go
        # to a binary cross-entropy, which requires them in [0, 1]. Clipping here rather than
        # editing the shared fixture, which exists to test the sequence-length crop and has no
        # reason to care.
        arrays['val_multilabel_values'] = [
            np.clip(values, 0.0, 1.0) for values in arrays['val_multilabel_values']
        ]
        dataset = MixedDataset(**arrays)
        save_dataset(dataset, os.path.join(fold_dir, partition))


def _write_configs(root, data_dir, model_dir):
    """Write the dataset config and the tuning base config.

    Args:
        root: Directory to write the configs into.
        data_dir: DATA_DIR for the dataset config.
        model_dir: MODEL_DIR for the experiment config.

    Returns:
        A tuple of (dataset config path, base config path).
    """
    dataset_config = {
        'DATA_DIR': data_dir,
        'VARIABLE_PROPERTIES_PATH': os.path.join(root, 'variable_properties.yaml'),
        'VALUED_FEATS': VALUED_FEATS,
        'EVENT_FEATS': EVENT_FEATS,
        'TEXT_FEATS': TEXT_FEATS,
        'STATIC_FEATS': STATIC_FEATS,
        'MAX_HISTORY_LEN_STEPS': MAX_HISTORY_LEN_STEPS,
    }
    dataset_config_path = os.path.join(root, 'dataset.yaml')
    with open(dataset_config_path, 'w') as f_out:
        yaml.dump(dataset_config, f_out)

    base_config = {
        'EXPERIMENT_NAME': 'itest_base',
        'BATCH_SIZE': 2,
        'USE_TEXT': True,
        'USE_HISTORICAL_RECORDS': True,
        'HISTORY_LEN_STEPS': None,
        'EPISODE_LEN_STEPS': None,
        'PREDICT_INDICATORS': False,
        'POSITION_ENCODING': 'additive',
        'VALUE_LADDER_P_MIN': 2.0,
        'VALUE_LADDER_P_MAX': 5000.0,
        'EVENT_LADDER_P_MIN': 2.0,
        'EVENT_LADDER_P_MAX': 500.0,
        'GENERATOR_ENCODER_D_MODEL': 32,
        'GENERATOR_ENCODER_N_HEADS': 2,
        'GENERATOR_ENCODER_N_ENCODER_BLOCKS': 1,
        'GENERATOR_ENCODER_DIM_FEEDFORWARD': 32,
        'GENERATOR_ENCODER_DROPOUT': 0.1,
        'GENERATOR_ENCODER_ACTIVATION': 'gelu',
        'GENERATOR_ENCODER_NORM': 'LayerNorm',
        'GENERATOR_ENCODER_NORM_FIRST': True,
        'DISCRIMINATOR_ENCODER_D_MODEL': 32,
        'DISCRIMINATOR_ENCODER_N_HEADS': 2,
        'DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS': 1,
        'DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD': 32,
        'DISCRIMINATOR_ENCODER_DROPOUT': 0.1,
        'DISCRIMINATOR_ENCODER_ACTIVATION': 'gelu',
        'DISCRIMINATOR_ENCODER_NORM': 'LayerNorm',
        'DISCRIMINATOR_ENCODER_NORM_FIRST': True,
        'THP_ENCODER_D_MODEL': 32,
        'THP_ENCODER_D_INNER': 32,
        'THP_ENCODER_N_LAYERS': 1,
        'THP_ENCODER_N_HEADS': 2,
        'THP_ENCODER_D_K': 16,
        'THP_ENCODER_D_V': 8,
        'THP_ENCODER_DROPOUT': 0.1,
        'THP_ENCODER_NORM_FIRST': True,
        'GENERATOR_D_MODEL': 32,
        'GENERATOR_DIM_FEEDFORWARD': 32,
        'DISCRIMINATOR_DIM_FEEDFORWARD': 32,
        'PREDICTOR_AGGREGATION_METHOD': 'mean',
        'MODEL_DIR': model_dir,
        'PRETRAIN_TOTAL_EPOCH': 1,
        'FINETUNE_TOTAL_EPOCH': 1,
        'FINETUNE_LEARNING_RATE': 0.0002,
        'FINETUNE_LEARNING_RATE_DECAY': 0.9,
        'DISC_LOSS_WEIGHT': 1.0,
        'USE_THP_PRED_LOSS': True,
        'THP_LOSS_MC_SAMPLES': 4,
        'THP_LOSS_NLL_WEIGHT': 0.01,
        'THP_PRED_LOSS_TYPE_WT': 1.0,
        'OBS_UNOBS_SAMPLE_RATIO': 10.0,
        'PRETRAIN_LEARNING_RATE': 0.002,
        'PRETRAIN_LEARNING_RATE_DECAY': 0.9,
        'CMPNT_MASK_RATIO': 0.25,
        'RECORD_MASK_RATIO': 0.15,
        'THP_PRED_LOSS_TIME_WT': 0.01,
    }
    base_config_path = os.path.join(root, 'base.yaml')
    with open(base_config_path, 'w') as f_out:
        yaml.dump(base_config, f_out)

    return dataset_config_path, base_config_path


def _run(command, cwd=None):
    """Run a subprocess with the repository on PYTHONPATH and return the completed process.

    Runs in a scratch directory rather than in the repository, because run_experiment.py writes
    ./log/<experiment>/ and ./checkpoints/<experiment>/ relative to the working directory. With
    cwd set to the repository those land in the real log/ and checkpoints/ trees and stay there
    after the test passes. The repository is reachable through PYTHONPATH, and the caller passes
    absolute script paths, so nothing depends on being run from the root.

    Args:
        command: Argument list. Script paths must be absolute.
        cwd: Working directory. Defaults to the module scratch directory.

    Returns:
        The CompletedProcess, with stdout and stderr merged into stdout.
    """
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = ''
    # Pinned to CPU rather than left to pick the best local device, which on an Apple Silicon
    # machine means MPS -- and MPS has an upstream bug that this model walks straight into.
    # `F.scaled_dot_product_attention` on MPS returns a tensor of the wrong width when the value
    # head dimension differs from the query/key one AND dropout_p is 0. The event encoder has
    # exactly that asymmetry (THP_ENCODER_D_K 128 against D_V 64, because only q and k are
    # rotated), so training passes -- dropout is on, which takes the math path -- and the first
    # validation pass dies in the output projection. CPU and CUDA are both correct.
    env['ACCELERATE_USE_CPU'] = '1'
    # A stale accelerate environment would make run_experiment.py refuse to start, which is the
    # behaviour under test elsewhere; here it would just be noise.
    for name in ('ACCELERATE_USE_FSDP', 'ACCELERATE_MIXED_PRECISION', 'ACCELERATE_CONFIG_FILE',
                 'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    return subprocess.run(
        command, cwd=cwd or _SCRATCH[0] or REPO_ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )


def script(name):
    """Absolute path to a repository entry point, so it can be run from anywhere.

    Args:
        name: File name at the repository root, e.g. 'run_experiment.py'.

    Returns:
        The absolute path.
    """
    return os.path.join(REPO_ROOT, name)


@pytest.fixture(scope='module')
def sweep(tmp_path_factory):
    """Build a synthetic fold, a spec and a manifest, and expand the trial configs.

    Returns:
        A dict of the paths the tests need.
    """
    root = str(tmp_path_factory.mktemp('phase2'))
    data_dir = os.path.join(root, 'data')
    model_dir = os.path.join(root, 'models')
    os.makedirs(data_dir, exist_ok=True)
    # Every subprocess runs here, so the runner's relative ./log and ./checkpoints land in the
    # temporary tree and go away with it.
    _SCRATCH[0] = root

    _write_variable_properties(os.path.join(root, 'variable_properties.yaml'))
    _write_fold(os.path.join(data_dir, 'fold0'))
    dataset_config, base_config = _write_configs(root, data_dir, model_dir)

    spec = {
        'SPEC_NAME': 'itest',
        'BASE_CONFIG': base_config,
        'DATASET_CONFIG': dataset_config,
        'OUTPUT_DIR': os.path.join(root, 'trials'),
        'MANIFEST': os.path.join(root, 'itest_manifest.yaml'),
        'FOLD': 'fold0',
        'ARMS': {'additive': {'POSITION_ENCODING': 'additive'},
                 'rope': {'POSITION_ENCODING': 'rope'}},
        'ALIASES': {'PRETRAIN_LEARNING_RATE': 'lr', 'CMPNT_MASK_RATIO': 'cmask'},
        'GRID': {
            'PRETRAIN_LEARNING_RATE': {'values': [0.002, 0.0002], 'select_on': 'pretrain'},
            'CMPNT_MASK_RATIO': {'values': [0.25, 0.5], 'select_on': 'mortality'},
        },
    }
    spec_path = os.path.join(root, 'itest_spec.yaml')
    with open(spec_path, 'w') as f_out:
        yaml.dump(spec, f_out)

    result = _run([sys.executable, script('generate_tuning_configs.py'), spec_path])
    assert result.returncode == 0, result.stdout

    with open(spec['MANIFEST'], 'r') as f_in:
        manifest = yaml.safe_load(f_in)

    return {
        'root': root,
        'dataset_config': dataset_config,
        'base_config': base_config,
        'spec': spec_path,
        'manifest_path': spec['MANIFEST'],
        'manifest': manifest,
        'model_dir': model_dir,
    }


def test_generator_expands_an_additive_sweep(sweep):
    """Two arms x (one centre + two non-default values) is six trials, four of them finetuned."""
    manifest = sweep['manifest']
    assert len(manifest['trials']) == 6, [t['name'] for t in manifest['trials']]

    finetuned = [t for t in manifest['trials'] if t['needs_finetune']]
    # The centre of each arm plus both CMPNT_MASK_RATIO values; the learning-rate trials are
    # ranked on pretraining loss and never finetuned.
    assert len(finetuned) == 4, [t['name'] for t in finetuned]

    centres = [t for t in manifest['trials'] if t['is_centre']]
    assert len(centres) == 2
    for centre in centres:
        covered = {c['hyperparameter'] for c in centre['covers']}
        assert covered == set(manifest['grid']), covered


def test_each_trial_config_sets_its_own_value(sweep):
    """The generated configs differ in exactly the arm and the one hyperparameter under test."""
    for trial in sweep['manifest']['trials']:
        with open(trial['config'], 'r') as f_in:
            config = yaml.safe_load(f_in)
        assert config['EXPERIMENT_NAME'] == trial['name']
        assert config['POSITION_ENCODING'] == trial['arm']
        if trial['is_centre']:
            for name, entry in sweep['manifest']['grid'].items():
                assert config[name] == entry['default'], (trial['name'], name)
        else:
            assert config[trial['hyperparameter']] == trial['value']
            others = set(sweep['manifest']['grid']) - {trial['hyperparameter']}
            for name in others:
                assert config[name] == sweep['manifest']['grid'][name]['default']


def test_trial_lookup_matches_the_manifest_order(sweep):
    """tuning_trial.py indexes the same lists the SLURM arrays will index."""
    for stage, expected in (('pretrain', 6), ('finetune', 4)):
        result = _run([sys.executable, script('tuning_trial.py'), sweep['manifest_path'],
                       '--stage', stage, '--count'])
        assert result.returncode == 0, result.stdout
        assert int(result.stdout.strip()) == expected

        result = _run([sys.executable, script('tuning_trial.py'), sweep['manifest_path'],
                       '--stage', stage, '--index', str(expected), '--field', 'name'])
        assert result.returncode == 1, 'an out-of-range array index must fail its own task'
        assert 'out of range' in result.stdout


def test_runner_rejects_a_tuning_grid(sweep):
    """A config carrying HYPERPARAMETERS_TO_TUNE is a grid, not a configuration."""
    grid_config = os.path.join(sweep['root'], 'grid.yaml')
    with open(sweep['base_config'], 'r') as f_in:
        config = yaml.safe_load(f_in)
    config['HYPERPARAMETERS_TO_TUNE'] = ['CMPNT_MASK_RATIO']
    config['CMPNT_MASK_RATIO'] = [0.25, 0.5]
    with open(grid_config, 'w') as f_out:
        yaml.dump(config, f_out)

    result = _run([sys.executable, script('run_experiment.py'), sweep['dataset_config'], grid_config,
                   '--folds', 'fold0', '--tasks', 'none'])
    assert result.returncode != 0
    assert 'HYPERPARAMETERS_TO_TUNE' in result.stdout


def test_runner_rejects_an_unknown_fold(sweep):
    """A mistyped fold fails in a second rather than after a pretrain."""
    result = _run([sys.executable, script('run_experiment.py'), sweep['dataset_config'],
                   sweep['base_config'], '--folds', 'fold9', '--tasks', 'none'])
    assert result.returncode != 0
    assert 'fold9' in result.stdout


@pytest.mark.slow
def test_pretrain_then_finetune_then_report_then_select(sweep):
    """The whole Phase 2 walk, end to end, on both arms.

    This is the test that matters. Everything else here checks one link; this one checks that
    the links connect -- in particular that the finetune stage finds the encoder weights the
    pretrain stage wrote, and that the reporting stage finds the evaluation YAMLs both wrote.
    """
    manifest = sweep['manifest']
    common = ['--folds', 'fold0', '--num_workers', '0']

    for trial in manifest['trials']:
        result = _run([sys.executable, script('run_experiment.py'), sweep['dataset_config'],
                       trial['config'], '--tasks', 'none'] + common)
        assert result.returncode == 0, f"{trial['name']}:\n{result.stdout[-4000:]}"

        pretrained_dir = os.path.join(sweep['model_dir'], trial['name'], 'fold0', 'pretrained')
        for name in ('value_encoder.pt', 'event_encoder.pt'):
            assert os.path.exists(os.path.join(pretrained_dir, name)), \
                f"{trial['name']} did not write {name}; the finetune stage loads it"

        evaluation = os.path.join(pretrained_dir, 'evaluation', 'evaluation_pretrained.yaml')
        assert os.path.exists(evaluation), \
            f"{trial['name']} recorded no pretraining loss, so it cannot be ranked on one"
        with open(evaluation, 'r') as f_in:
            data = yaml.safe_load(f_in)
        assert 'Optimization_Loss' in data['val_losses']
        # The recorded hyperparameters are what makes a result interpretable without the
        # manifest that produced it.
        assert data['hyperparameters']['POSITION_ENCODING'] == trial['arm']

    for trial in [t for t in manifest['trials'] if t['needs_finetune']]:
        result = _run([sys.executable, script('run_experiment.py'), sweep['dataset_config'],
                       trial['config'], '--tasks', 'mortality'] + common)
        assert result.returncode == 0, f"{trial['name']}:\n{result.stdout[-4000:]}"
        # Pretrained weights were already on disk, so this run must have skipped pretraining.
        assert 'skipping pretraining' in result.stdout, \
            f"{trial['name']} re-pretrained instead of reusing its own weights"

        evaluation = os.path.join(sweep['model_dir'], trial['name'], 'fold0', 'mortality',
                                  'evaluation', 'evaluation_mortality.yaml')
        assert os.path.exists(evaluation), f"{trial['name']} wrote no mortality evaluation"
        with open(evaluation, 'r') as f_in:
            data = yaml.safe_load(f_in)
        assert data['validation_scores'] is not None, \
            'without validation scores the downstream-selected hyperparameters cannot be ranked'
        assert 'AUPRC' in data['validation_scores']

    tables_dir = os.path.join(sweep['root'], 'tables')
    result = _run([sys.executable, script('report_tuning_results.py'), sweep['manifest_path'],
                   '--tables_dir', tables_dir, '--require_complete'])
    assert result.returncode == 0, result.stdout
    for name in ('itest_tuning.docx', 'itest_tuning.csv'):
        assert os.path.exists(os.path.join(tables_dir, name)), f'{name} was not written'

    assembled_path = os.path.join(sweep['root'], 'assembled.yaml')
    result = _run([sys.executable, script('select_tuned_hyperparameters.py'), sweep['manifest_path'],
                   '--output', assembled_path])
    assert result.returncode == 0, result.stdout

    with open(assembled_path, 'r') as f_in:
        assembled = yaml.safe_load(f_in)
    for name, entry in manifest['grid'].items():
        assert name in assembled, f'{name} is missing from the assembled config'
        assert assembled[name] in entry['values'], \
            f'{name} was assembled as {assembled[name]!r}, which is not in its grid'
    assert assembled['POSITION_ENCODING'] in manifest['arms']
    # The assembled config has to be runnable, which means carrying everything the arm's centre
    # carried and not just the tuned values.
    assert 'THP_ENCODER_D_MODEL' in assembled
    assert 'MODEL_DIR' in assembled

    selection_path = os.path.join(sweep['root'], 'itest_selection.yaml')
    assert os.path.exists(selection_path)
    with open(selection_path, 'r') as f_in:
        decision = yaml.safe_load(f_in)
    assert decision['caveats'], 'the selection record must carry the additive-sweep caveat'


@pytest.mark.slow
def test_limit_episodes_truncates_every_partition(sweep):
    """--limit_episodes is what keeps the GPU smoke test inside its time limit."""
    with open(sweep['base_config'], 'r') as f_in:
        config = yaml.safe_load(f_in)
    config['EXPERIMENT_NAME'] = 'itest_limited'
    limited_path = os.path.join(sweep['root'], 'limited.yaml')
    with open(limited_path, 'w') as f_out:
        yaml.dump(config, f_out)

    result = _run([sys.executable, script('run_experiment.py'), sweep['dataset_config'], limited_path,
                   '--folds', 'fold0', '--tasks', 'none', '--num_workers', '0',
                   '--limit_episodes', '2'])
    assert result.returncode == 0, result.stdout[-4000:]
    assert 'SMOKE TEST' in result.stdout


@pytest.mark.slow
def test_multi_task_run_keeps_one_model_registered_per_stage(sweep):
    """A Phase 4 style run -- pretrain plus more than one task in one process -- must not let
    the Accelerator accumulate prepared models.

    ``Accelerator.prepare`` APPENDS to an internal registry and ``save_state`` writes every
    entry in it. Without an unregister between stages, a checkpoint taken ten epochs into the
    second task carries the pretraining model and the first task's model as well, and a
    resuming process -- which reaches that stage having prepared only the model it needs --
    loads the wrong entry into it. It surfaces as a size mismatch rather than as silent
    corruption, but it kills the resume.

    The accelerate runner discards the whole Accelerator between stages, which hides this. This
    runner keeps one for the process, so the unregister has to be explicit.

    Two tasks are enough: the bug needs a stage that prepares a model after another already
    has. Phenotype is left out because it needs a phenotyping listfile the synthetic fold does
    not carry.
    """
    with open(sweep['base_config'], 'r') as f_in:
        config = yaml.safe_load(f_in)
    config['EXPERIMENT_NAME'] = 'itest_multitask'
    config_path = os.path.join(sweep['root'], 'multitask.yaml')
    with open(config_path, 'w') as f_out:
        yaml.dump(config, f_out)

    probe = os.path.join(sweep['root'], 'registry_probe.py')
    with open(probe, 'w') as f_out:
        f_out.write(
            "import sys\n"
            "import accelerate\n"
            "_prepare = accelerate.Accelerator.prepare\n"
            "_worst = [0]\n"
            "def prepare(self, *args, **kwargs):\n"
            "    out = _prepare(self, *args, **kwargs)\n"
            "    _worst[0] = max(_worst[0], len(self._models))\n"
            "    return out\n"
            "accelerate.Accelerator.prepare = prepare\n"
            "import atexit\n"
            "atexit.register(lambda: print(f'MAX_REGISTERED_MODELS={_worst[0]}'))\n"
            "sys.argv = sys.argv[1:]\n"
            "import runpy\n"
            f"runpy.run_path({script('run_experiment.py')!r}, run_name='__main__')\n"
        )

    result = _run([sys.executable, probe, script('run_experiment.py'),
                   sweep['dataset_config'], config_path,
                   '--folds', 'fold0', '--tasks', 'mortality', 'length_of_stay',
                   '--num_workers', '0'])
    assert result.returncode == 0, result.stdout[-4000:]

    marker = [line for line in result.stdout.splitlines()
              if line.startswith('MAX_REGISTERED_MODELS=')]
    assert marker, f'the probe did not report:\n{result.stdout[-2000:]}'
    worst = int(marker[-1].split('=')[1])
    assert worst == 1, (
        f'the Accelerator held {worst} prepared models at once. Every stage must unregister '
        f'its model before the next one prepares, or a checkpoint written in a later stage '
        f'cannot be resumed.'
    )


def test_measure_spans_ignores_padding():
    """The span measurement that checks the ladder must not treat padded zeros as records.

    Padded timestamps are stored as 0.0, which is a legitimate admission-relative time. Reading
    the span off the raw array would report the distance from an imaginary record at hour zero
    and inflate the measured Delta_max, which is the number the whole ladder derivation rests on.
    """
    from test_phase2_pipeline import measure_spans

    times = np.array([[0.0, 0.0, 100.0, 110.0], [0.0, 0.0, 0.0, 5.0]], dtype=np.float32)
    masks = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    measured = measure_spans(times, masks)
    assert measured['max_span'] == pytest.approx(10.0)
    assert measured['max_step_gap'] == pytest.approx(10.0)
    assert measured['min_step_gap'] == pytest.approx(10.0)
    assert measured['n_episodes_with_records'] == 2


def test_generated_test_spec_survives_a_relative_work_dir(tmp_path, monkeypatch):
    """write_test_spec must produce a spec whose paths load_spec() can resolve.

    `load_spec` resolves a relative BASE_CONFIG, DATASET_CONFIG, OUTPUT_DIR or MANIFEST relative
    to the spec file's own directory, which is what lets the real `phase2_spec.yaml` name
    `phase2_base.yaml` as a bare filename beside it. `slurm_test_phase2.sh` defaults `WORK_DIR`
    to a relative path, so a relative BASE_CONFIG is joined onto the spec dir and doubled.

    The probe drives the writer with a relative work_dir and asserts every path round-trips.
    """
    import test_phase2_pipeline
    from hp_tuning.spec import load_spec

    # A relative work_dir is the whole point; resolve it against tmp_path, not the repo.
    monkeypatch.chdir(tmp_path)
    os.makedirs('log/test_phase2_relative', exist_ok=True)

    dataset_config = os.path.join(
        REPO_ROOT, 'TransEHR2', 'configs', 'datasets', 'mimic4.yaml')
    base_config_path = os.path.join(
        REPO_ROOT, 'TransEHR2', 'configs', 'experiments', 'tuning', 'phase2_base.yaml')
    with open(base_config_path) as fh:
        base_config = yaml.safe_load(fh)

    args = argparse.Namespace(
        dataset_config=dataset_config,
        fold='fold0',
        arms=['additive', 'rope'],
        epochs=1,
    )
    spec_path = test_phase2_pipeline.write_test_spec(
        args, 'log/test_phase2_relative', base_config
    )

    spec = load_spec(spec_path)

    assert os.path.exists(spec['BASE_CONFIG']), spec['BASE_CONFIG']
    # The doubling is the signature of the bug: the spec dir appearing twice in one path.
    for key in ('BASE_CONFIG', 'DATASET_CONFIG', 'OUTPUT_DIR', 'MANIFEST'):
        value = spec[key]
        assert os.path.isabs(value), '%s is not absolute: %s' % (key, value)
        assert value.count('log/test_phase2_relative') <= 1, (
            '%s has the work dir doubled: %s' % (key, value))

    # MODEL_DIR is read out of the base config, so it has to be absolute there too or the
    # trials write their evaluation YAMLs somewhere the reporter will not look.
    with open(spec['BASE_CONFIG']) as fh:
        written_base = yaml.safe_load(fh)
    assert os.path.isabs(written_base['MODEL_DIR']), written_base['MODEL_DIR']


def test_deadline_helpers_stop_before_the_allocation_ends(monkeypatch):
    """`budget_allows` gates each trial on the time left in the SLURM allocation.

    A job cut off at its limit never prints the summary or the `--time` recommendation, so the
    trial loops stop while a reserve remains for the reporting and selection stages.
    """
    import test_phase2_pipeline as gate

    # No SLURM environment: never blocks, so local runs are unaffected.
    monkeypatch.delenv('SLURM_JOB_END_TIME', raising=False)
    assert gate.seconds_until_job_ends() is None
    assert gate.budget_allows(9_999_999, 420)[0] is True

    # A malformed value must not crash the test harness mid-sweep.
    monkeypatch.setenv('SLURM_JOB_END_TIME', 'not-a-timestamp')
    assert gate.seconds_until_job_ends() is None
    assert gate.budget_allows(9_999_999, 420)[0] is True

    # Thirty minutes left, seven reserved: a ten-minute trial fits, a thirty-minute one does not.
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(time.time() + 30 * 60))
    assert gate.budget_allows(10 * 60, 420)[0] is True
    allowed, reason = gate.budget_allows(30 * 60, 420)
    assert allowed is False
    assert 'left in the allocation' in reason

    # An unknown estimate cannot be judged, so the first trial always runs.
    assert gate.budget_allows(None, 420)[0] is True
