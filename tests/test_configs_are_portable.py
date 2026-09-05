"""Tracked configs must not carry paths belonging to one machine.

`load_spec` resolves a spec's relative paths against the spec's own directory, and
`write_manifest` then records the resolved absolutes. That is correct at runtime and wrong in
version control: a manifest and its generated trial configs were committed from a laptop, so a
sweep launched from them on the cluster read `/Users/...` for its dataset config. They are also
a snapshot of the base config, and went stale the moment a bound in it was re-derived.

Generated sweep output is now ignored rather than tracked. These probes hold that, and catch any
other tracked config that grows a host-specific path.
"""

import os
import re
import subprocess

import pytest
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A leading slash that is not the repository itself. Cluster paths are as unportable as laptop
# ones; MODEL_DIR and DATA_DIR are the deliberate exceptions, since they name storage that has
# no meaning relative to a checkout.
ABSOLUTE_PATH = re.compile(r'(?<![\w.])/(?:[\w.-]+/)+[\w.-]+')
ALLOWED_KEYS = {'MODEL_DIR', 'DATA_DIR', 'VARIABLE_PROPERTIES_PATH'}


# A decision record names the manifest and the trials it read, on the machine that ran them.
# That is what it is for, so it is not held to the portability rule the configs are.
RECORD_SUFFIXES = ('_cell.yaml', '_selection.yaml')


def tracked_configs():
    """Every tracked YAML under TransEHR2/configs/ that is a config rather than a record."""
    listed = subprocess.run(
        ['git', 'ls-files', 'TransEHR2/configs/*.yaml',
         'TransEHR2/configs/**/*.yaml'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted({path for path in listed if not path.endswith(RECORD_SUFFIXES)})


def absolute_values(node, path=''):
    """Yield (key path, value) for every string in `node` that looks like an absolute path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from absolute_values(value, f'{path}.{key}' if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from absolute_values(value, f'{path}[{index}]')
    elif isinstance(node, str) and node.startswith('/') and ABSOLUTE_PATH.match(node):
        yield path, node


def test_there_are_tracked_configs_to_check():
    """A glob that silently matches nothing would make every probe below vacuous."""
    assert len(tracked_configs()) >= 5


@pytest.mark.parametrize('relative_path', tracked_configs())
def test_no_tracked_config_carries_a_host_specific_path(relative_path):
    with open(os.path.join(REPO_ROOT, relative_path)) as f_in:
        config = yaml.safe_load(f_in)
    offenders = [
        (key, value) for key, value in absolute_values(config or {})
        if key.split('.')[-1].split('[')[0] not in ALLOWED_KEYS
    ]
    assert not offenders, (
        f'{relative_path} carries absolute paths that belong to whichever machine wrote it: '
        f'{offenders}. Generated sweep output should not be tracked; regenerate it with '
        f'generate_tuning_configs.py on the machine that will run the sweep.'
    )


@pytest.mark.parametrize('pattern', [
    'TransEHR2/configs/experiments/tuning/phase2/',
    'TransEHR2/configs/experiments/tuning/phase3/',
    'TransEHR2/configs/experiments/tuning/phase2_manifest.yaml',
])
def test_generated_sweep_output_is_not_tracked(pattern):
    listed = subprocess.run(['git', 'ls-files', pattern],
                            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert not listed.stdout.strip(), (
        f'{pattern} is tracked again. It is written by generate_tuning_configs.py from the '
        f'spec, carries the writing machine\'s absolute paths, and snapshots the base config.'
    )


def tracked_specs():
    """Every tracked tuning spec, by repo-relative path."""
    listed = subprocess.run(
        ['git', 'ls-files', 'TransEHR2/configs/experiments/tuning/*_spec.yaml'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in listed.stdout.split('\n') if line]


@pytest.mark.parametrize('path', tracked_specs())
def test_the_spec_names_its_paths_relatively(path):
    """The spec is the tracked half of the pair, so its paths have to travel."""
    with open(os.path.join(REPO_ROOT, path)) as f_in:
        spec = yaml.safe_load(f_in)
    for key in ('BASE_CONFIG', 'DATASET_CONFIG', 'OUTPUT_DIR', 'MANIFEST'):
        assert not os.path.isabs(spec[key]), f'{path}: {key} is absolute'


def test_there_are_specs_to_check():
    """Guard the parametrization: an empty list would make the test above vacuous."""
    assert tracked_specs()
