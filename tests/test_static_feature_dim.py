"""Regression probe: the static width the model builds for must match the width on disk.

`MixedClassifier` concatenates the value embedding with `static_data` before its feedforward
layer, so the layer's input width is `d_model + static_width`. The entry points used to pass
`len(STATIC_FEATS)` as that width -- a feature *count*. The extraction writes
`sum(static_feat_dims)`, and since 75c39e8 ("Fixed categorical/ordinal feature encoding scheme,
now actually one-hot") a categorical static occupies `size` columns rather than one.

With the shipped MIMIC-IV config that is Age (numeric, 1) + Gender (categorical, 3) = 4 columns
against a declared 2, so the first forward pass dies with

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (110000x260 and 258x256)

  110000 = batch 200 x 550 timesteps,  260 = 256 + 4 (data),  258 = 256 + 2 (model)

Arrays extracted before 75c39e8 were the old width and matched, which is why this only surfaced
on a re-extraction. These probes assert the model-side and extraction-side derivations agree,
and that the count is not a valid stand-in for the width.
"""

import os
import re
import subprocess
import yaml

import pytest

from TransEHR2.constants import MAX_TOKEN_LENGTH
from TransEHR2.data.preprocessing import compute_static_feat_dims


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `n_static_features=len(STATIC_FEATS)` and friends -- an argument, not prose about one.
ARGUMENT_PATTERN = re.compile(r'=\s*len\(STATIC_FEATS\)')
DATASET_CONFIG = os.path.join(REPO_ROOT, 'TransEHR2', 'configs', 'datasets', 'mimic4.yaml')
VARIABLE_PROPERTIES = os.path.join(REPO_ROOT, 'data', 'variable_properties.yaml')


@pytest.fixture(scope='module')
def shipped_config():
    with open(DATASET_CONFIG) as fh:
        dataset_config = yaml.safe_load(fh)
    with open(VARIABLE_PROPERTIES) as fh:
        variable_properties = yaml.safe_load(fh)
    return dataset_config, variable_properties


def test_one_hot_static_is_wider_than_the_feature_count(shipped_config):
    """The bug in one line: for the shipped config, count != width."""
    dataset_config, variable_properties = shipped_config
    static_feats = dataset_config['STATIC_FEATS']

    width = sum(compute_static_feat_dims(
        variable_properties, static_feats, MAX_TOKEN_LENGTH
    ))
    assert width != len(static_feats), (
        'this config no longer distinguishes the two derivations, so it cannot defend against '
        'the confusion; point the probe at a config with a categorical static'
    )
    assert width == 4 and len(static_feats) == 2


def test_helper_matches_the_extraction(shipped_config):
    """`_get_tensor_dimensions` must route through the same helper the model side uses."""
    from TransEHR2.data import preprocessing
    import inspect

    source = inspect.getsource(preprocessing._get_tensor_dimensions)
    assert 'compute_static_feat_dims(' in source, (
        'the extraction has stopped using the shared helper, so the two derivations can drift '
        'apart again -- which is exactly the failure this probe exists to prevent'
    )


def test_categorical_static_contributes_its_size(shipped_config):
    _, variable_properties = shipped_config
    dims = compute_static_feat_dims(
        variable_properties, ['Age', 'Gender'], MAX_TOKEN_LENGTH
    )
    assert dims == [1, 3]


def test_text_static_contributes_the_token_length():
    variable_properties = {
        'Note': {'type': 'text', 'size': 1},
        'Age': {'type': 'numeric', 'size': 1},
    }
    dims = compute_static_feat_dims(variable_properties, ['Note', 'Age'], 8192)
    assert dims == [8192, 1]


def test_entry_points_do_not_pass_the_feature_count():
    """Guard every entry point at once: none of them may size statics by count again.

    Enumerates what git tracks rather than listing filenames. An earlier version of this probe
    named four files and so never looked at `run_experiment.py` -- which lives only on the
    branches carrying the single-GPU runner, and is the one entry point the Phase 2 smoke test
    actually invokes. It kept the bug through two rounds of fixes while this suite stayed green.

    `git ls-files` rather than `os.walk` because the working tree also holds ignored local
    copies of the entry points; those are nobody's deliverable, and walking the filesystem makes
    the probe's verdict depend on which stray files a given checkout happens to have.
    """
    tracked = subprocess.run(
        ['git', 'ls-files', '*.py'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert tracked, 'git ls-files returned nothing; this probe cannot verify anything'

    offenders = []
    for relative in tracked:
        # This probe names the anti-pattern in its own prose and its own assertion.
        if relative == 'tests/test_static_feature_dim.py':
            continue
        path = os.path.join(REPO_ROOT, relative)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            try:
                lines = fh.readlines()
            except UnicodeDecodeError:
                continue
        for lineno, line in enumerate(lines, 1):
            # Match an actual argument or assignment, not a mention. The name appears in
            # comments and docstrings that exist precisely to warn against it, and a probe
            # that trips on its own warnings teaches people to delete the warnings.
            if not ARGUMENT_PATTERN.search(line):
                continue
            offenders.append('%s:%d: %s' % (relative, lineno, line.strip()))

    assert not offenders, (
        'static dimensions must come from compute_static_feat_dims(), not a feature count:\n'
        + '\n'.join(offenders)
    )
