"""Probes for inverse-prevalence weighting of the mortality loss.

The weight exists so that a classifier is not rewarded for predicting the majority class
everywhere. It is deliberately confined to single-label tasks: applied per label under a
multi-label task it expresses a preference for correctly classifying rare labels over common
ones, which is a modelling stance rather than a correction for imbalance.
"""

import ast
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TransEHR2.utils import describe_class_weights, positive_class_weight


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, mortality=None, phenotype=None):
        self.mortality = mortality
        self.phenotype = phenotype


def binary(n, prevalence, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < prevalence).astype(np.float32)


def test_the_weight_is_the_ratio_of_negatives_to_positives():
    labels = binary(50_000, 0.1)
    weight = positive_class_weight(Dataset(mortality=labels), 'mortality')
    p = labels.mean()
    assert weight.shape == (1,)
    assert weight.item() == pytest.approx((1 - p) / p)


def test_the_weight_equalises_what_the_two_classes_contribute():
    labels = binary(50_000, 0.1)
    weight = positive_class_weight(Dataset(mortality=labels), 'mortality')
    targets = torch.from_numpy(labels)[:, None]
    logits = torch.zeros_like(targets)

    def split(pos_weight):
        per = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none')
        return (per * targets).sum().item(), (per * (1 - targets)).sum().item()

    unweighted_pos, unweighted_neg = split(None)
    weighted_pos, weighted_neg = split(weight)
    assert unweighted_pos / unweighted_neg < 0.2      # positives barely register
    assert weighted_pos / weighted_neg == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize('task', ['length_of_stay', 'los', 'not_a_task'])
def test_a_task_that_is_not_binary_gets_no_weight(task):
    assert positive_class_weight(Dataset(mortality=binary(500, 0.1)), task) is None


@pytest.mark.parametrize('labels', [np.ones(500, np.float32), np.zeros(500, np.float32)])
def test_a_label_with_only_one_class_keeps_a_weight_of_one(labels):
    """The ratio is undefined; a large substitute would let an unlearnable label dominate."""
    weight = positive_class_weight(Dataset(mortality=labels), 'mortality')
    assert weight.tolist() == [1.0]


def test_a_multilabel_task_yields_one_weight_per_label():
    """The mechanism is general even though the default does not use it."""
    rng = np.random.default_rng(1)
    phenotype = (rng.random((20_000, 25)) < rng.uniform(0.02, 0.4, 25)).astype(np.float32)
    weight = positive_class_weight(Dataset(phenotype=phenotype), 'phenotype')
    assert weight.shape == (25,)


def default_of(function_name, parameter):
    """The default in the source, read without importing the module."""
    tree = ast.parse(open(os.path.join(ROOT, 'TransEHR2', 'routines_accelerate.py')).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = node.args.args + node.args.kwonlyargs
            defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                        + list(node.args.defaults)) + list(node.args.kw_defaults)
            for arg, default in zip(args, defaults):
                if arg.arg == parameter:
                    return ast.literal_eval(default)
    raise AssertionError(f'{function_name} has no parameter {parameter}')


def test_phenotype_is_not_weighted_by_default():
    assert default_of('finetune_model', 'pos_weight_tasks') == ('mortality',)


def test_the_description_reports_the_prevalence_it_measured():
    labels = binary(50_000, 0.1)
    line = describe_class_weights(
        positive_class_weight(Dataset(mortality=labels), 'mortality'),
        'mortality', Dataset(mortality=labels))
    assert 'pos_weight' in line and '50,000 episodes' in line


def test_an_unweighted_task_says_so():
    assert 'unweighted' in describe_class_weights(None, 'phenotype', Dataset())
