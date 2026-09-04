"""Reading tuning trial results back off disk, and ranking them.

Trials write the same two files every experiment writes -- ``evaluation_pretrained.yaml`` from
the pretraining stage and ``evaluation_<task>.yaml`` from each finetune -- so nothing here is
specific to tuning except the manifest that says which trial stands for which grid cell.

The ranking rule is per-hyperparameter, not global. ``SELECTION_CRITERIA`` in
:mod:`hp_tuning.spec` records which criterion each hyperparameter uses and why; this module
just applies it. The accelerate tuner ranked every hyperparameter on minimum
``Optimization_Loss``, which is wrong for the three that rescale that loss: a higher mask ratio
makes the reconstruction task harder, so its pretraining loss is worse by construction and
"lowest loss wins" would pick the easiest task rather than the best model.
"""

import math
import os

from typing import Any, Dict, List, Optional

import yaml

from hp_tuning.spec import SELECTION_CRITERIA


class TrialResult:
    """The outcome of one trial under one selection criterion.

    Attributes:
        trial: The manifest entry for the trial.
        criterion: The selection criterion name, 'pretrain' or 'mortality'.
        value: The metric value, or None if the trial has not produced one.
        status: 'ok', 'pending' (no evaluation file yet), 'unreadable' (file present but the
            metric is not in it) or 'nan' (metric present but not a finite number).
        detail: Human-readable explanation for a non-'ok' status, else the path read.
        scores: The full metric block, for the report's extra columns.
        grid_value: The grid value or arm this result stands for. Set by the ranking
            functions, which know the grid; None when a result is read on its own.
    """

    __slots__ = ('trial', 'criterion', 'value', 'status', 'detail', 'scores', 'grid_value')

    def __init__(self, trial, criterion, value=None, status='pending', detail='', scores=None):
        self.trial = trial
        self.criterion = criterion
        self.value = value
        self.status = status
        self.detail = detail
        self.scores = scores or {}
        self.grid_value = None

    @property
    def name(self) -> str:
        """The trial's experiment name."""
        return self.trial['name']

    @property
    def is_usable(self) -> bool:
        """Whether this result can take part in a ranking."""
        return self.status == 'ok'


def evaluation_path(manifest: Dict[str, Any], trial: Dict[str, Any], criterion: str) -> str:
    """Where a trial's evaluation YAML for one criterion lives.

    Args:
        manifest: A loaded manifest.
        trial: One of its trials.
        criterion: 'pretrain' or 'mortality'.

    Returns:
        The path, which may not exist yet.
    """
    spec = SELECTION_CRITERIA[criterion]
    base = os.path.join(manifest['model_dir'], trial['name'], manifest['fold'])
    if spec['task'] == 'pretrain':
        return os.path.join(base, 'pretrained', 'evaluation', spec['evaluation_file'])
    return os.path.join(base, spec['task'], 'evaluation', spec['evaluation_file'])


def read_result(manifest: Dict[str, Any], trial: Dict[str, Any], criterion: str) -> TrialResult:
    """Read one trial's metric under one criterion.

    A missing file is reported as 'pending' rather than raised, because the normal state of a
    sweep in progress is that most files are missing and the report has to be readable anyway.

    Args:
        manifest: A loaded manifest.
        trial: One of its trials.
        criterion: 'pretrain' or 'mortality'.

    Returns:
        The result, whose ``status`` says whether the value can be trusted.
    """
    spec = SELECTION_CRITERIA[criterion]
    path = evaluation_path(manifest, trial, criterion)

    if not os.path.exists(path):
        return TrialResult(trial, criterion, status='pending', detail=f'not yet written: {path}')

    try:
        with open(path, 'r') as f_in:
            data = yaml.safe_load(f_in)
    except Exception as error:
        return TrialResult(trial, criterion, status='unreadable',
                           detail=f'{type(error).__name__} reading {path}: {error}')

    if not isinstance(data, dict):
        return TrialResult(trial, criterion, status='unreadable',
                           detail=f'{path} does not contain a mapping')

    block = data.get(spec['block'])
    if block is None:
        # Reachable when a finetune was skipped because a finetuned model already existed:
        # run_experiment.py has no validation scores to record in that case.
        return TrialResult(
            trial, criterion, status='unreadable', scores={},
            detail=(f"{path} has no {spec['block']!r} block. If the finetune was skipped "
                    f"because a model already existed, re-run it with --force_finetune.")
        )

    if spec['metric'] not in block:
        return TrialResult(
            trial, criterion, status='unreadable', scores=block,
            detail=f"{path}: {spec['block']} has no {spec['metric']!r}; "
                   f"present keys are {sorted(block)}"
        )

    value = block[spec['metric']]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return TrialResult(trial, criterion, status='nan', scores=block,
                           detail=f"{path}: {spec['metric']} is {value!r}, not a number")
    if not math.isfinite(value):
        # Both routines substitute inf or nan when predictions go bad rather than crashing, so
        # a non-finite metric is a real training outcome and must not silently win a ranking.
        return TrialResult(trial, criterion, status='nan', scores=block,
                           detail=f"{path}: {spec['metric']} is {value}")

    return TrialResult(trial, criterion, value=value, status='ok', detail=path, scores=block)


def collect(manifest: Dict[str, Any], criterion: Optional[str] = None) -> Dict[str, TrialResult]:
    """Read every trial's result under a criterion.

    Args:
        manifest: A loaded manifest.
        criterion: 'pretrain' or 'mortality'. Defaults to reading each trial under the
            criterion its own hyperparameter uses, with centres read under both.

    Returns:
        A dict keyed ``f'{criterion}/{trial_name}'`` so that a centre read under both criteria
        does not collide with itself.
    """
    criteria = [criterion] if criterion else sorted(SELECTION_CRITERIA)
    results = {}
    for name in criteria:
        for trial in manifest['trials']:
            if name != 'pretrain' and not trial['needs_finetune']:
                continue
            results[f'{name}/{trial["name"]}'] = read_result(manifest, trial, name)
    return results


def trial_for_value(
        manifest: Dict[str, Any],
        arm: str,
        hyperparameter: str,
        value: Any
) -> Optional[Dict[str, Any]]:
    """Find the trial that ran one hyperparameter at one value on one arm.

    For a non-default value this is the trial dedicated to it; for the default it is the arm's
    centre, which covers every hyperparameter's default at once. Matching goes through the
    manifest's ``covers`` list rather than through the trial's own ``hyperparameter`` field
    precisely so that the centre is found the same way as everything else.

    Args:
        manifest: A loaded manifest.
        arm: The encoding arm.
        hyperparameter: The hyperparameter name.
        value: The value to look for.

    Returns:
        The trial, or None if the grid does not contain that cell.
    """
    for trial in manifest['trials']:
        if trial['arm'] != arm:
            continue
        for covered in trial['covers']:
            if covered['hyperparameter'] == hyperparameter and covered['value'] == value:
                return trial
    return None


def rank_hyperparameter(
        manifest: Dict[str, Any],
        arm: str,
        hyperparameter: str
) -> Dict[str, Any]:
    """Rank the tested values of one hyperparameter on one arm.

    Args:
        manifest: A loaded manifest.
        arm: The encoding arm.
        hyperparameter: The hyperparameter to rank.

    Returns:
        A dict with:
            ``criterion``, ``metric``, ``direction``: how the ranking was done.
            ``results``: one :class:`TrialResult` per grid value, in grid order, each with an
                extra ``grid_value`` attribute set.
            ``best``: the winning result, or None if no value produced a usable metric.
            ``complete``: whether every grid value produced a usable metric. A winner picked
                from a partial set is a winner among the trials that happened to finish, which
                is not the same claim.
    """
    grid_entry = manifest['grid'][hyperparameter]
    criterion = grid_entry['select_on']
    spec = SELECTION_CRITERIA[criterion]

    results = []
    for value in grid_entry['values']:
        trial = trial_for_value(manifest, arm, hyperparameter, value)
        if trial is None:
            result = TrialResult(
                {'name': f'<missing {hyperparameter}={value}>', 'arm': arm},
                criterion, status='pending',
                detail=f'no trial in the manifest covers {hyperparameter}={value} on {arm}'
            )
        else:
            result = read_result(manifest, trial, criterion)
        result.grid_value = value
        results.append(result)

    usable = [r for r in results if r.is_usable]
    if usable:
        chooser = min if spec['direction'] == 'min' else max
        best = chooser(usable, key=lambda r: r.value)
    else:
        best = None

    return {
        'hyperparameter': hyperparameter,
        'arm': arm,
        'criterion': criterion,
        'metric': spec['metric'],
        'direction': spec['direction'],
        'results': results,
        'best': best,
        'complete': len(usable) == len(results),
    }


def rank_cells(
    manifest: Dict[str, Any],
    arm: str,
    criterion: Optional[str] = None
) -> Dict[str, Any]:
    """Rank whole grid cells on one arm, for a factorial sweep.

    A factorial exists because the hyperparameters interact, which is exactly the case in
    which the best combination is not the combination of individual bests. Ranking whole cells
    is what keeps that interaction in the answer; ranking coordinates would discard it.

    Args:
        manifest: A loaded manifest, from a sweep whose design is ``factorial``.
        arm: The encoding arm.
        criterion: Which criterion to rank on. Defaults to the one the grid shares, which a
            factorial spec is validated to have.

    Returns:
        A dict with:
            ``criterion``, ``metric``, ``direction``: how the ranking was done.
            ``results``: one :class:`TrialResult` per cell, in manifest order, each with
                ``grid_value`` set to that cell's assignment.
            ``best``: the winning cell's result, or None if no cell produced a usable metric.
            ``complete``: whether every cell produced a usable metric.

    Raises:
        ValueError: If the manifest is not a factorial sweep.
    """
    if manifest.get('design') != 'factorial':
        raise ValueError(
            f"rank_cells is for a factorial sweep; {manifest['spec_name']} is "
            f"{manifest.get('design', 'additive')!r}. Use rank_hyperparameter instead."
        )
    if criterion is None:
        criterion = next(iter(manifest['grid'].values()))['select_on']
    spec = SELECTION_CRITERIA[criterion]

    results = []
    for trial in manifest['trials']:
        if trial['arm'] != arm or trial.get('is_extra') or not trial.get('cell'):
            continue
        result = read_result(manifest, trial, criterion)
        result.grid_value = trial['cell']
        results.append(result)

    usable = [r for r in results if r.is_usable]
    chooser = min if spec['direction'] == 'min' else max
    return {
        'arm': arm,
        'criterion': criterion,
        'metric': spec['metric'],
        'direction': spec['direction'],
        'results': results,
        'best': chooser(usable, key=lambda r: r.value) if usable else None,
        'complete': bool(results) and len(usable) == len(results),
    }


def rank_all(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank every hyperparameter on every arm.

    Args:
        manifest: A loaded manifest.

    Returns:
        One ranking dict per (arm, hyperparameter), arms in manifest order.
    """
    return [
        rank_hyperparameter(manifest, arm, hyperparameter)
        for arm in manifest['arms']
        for hyperparameter in manifest['grid']
    ]


def compare_arms(manifest: Dict[str, Any], criterion: str = 'mortality') -> Dict[str, Any]:
    """Compare the encoding arms head to head at their shared centre.

    The arms differ only in ``POSITION_ENCODING``, so their centres are the one pair of runs
    that isolates the encoding: identical data, identical hyperparameters, identical frozen
    ladder, one rotating the gap into the attention scores and one adding it to the embedding.

    The revision plan flags the weakness of deciding here rather than at each arm's assembled
    optimum, and the conclusion is only as good as that assumption. It is recorded, not hidden.

    Args:
        manifest: A loaded manifest.
        criterion: Which criterion to compare on. Defaults to the downstream task.

    Returns:
        A dict with ``criterion``, ``metric``, ``direction``, ``results`` (one per arm) and
        ``best`` (the winning result, or None).
    """
    spec = SELECTION_CRITERIA[criterion]
    results = []
    for arm in manifest['arms']:
        centre = next(
            (t for t in manifest['trials'] if t['arm'] == arm and t['is_centre']), None
        )
        if centre is None:
            continue
        result = read_result(manifest, centre, criterion)
        result.grid_value = arm
        results.append(result)

    usable = [r for r in results if r.is_usable]
    chooser = min if spec['direction'] == 'min' else max
    return {
        'criterion': criterion,
        'metric': spec['metric'],
        'direction': spec['direction'],
        'results': results,
        'best': chooser(usable, key=lambda r: r.value) if usable else None,
        'complete': len(usable) == len(results),
    }


def progress(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Count how much of the sweep has produced results.

    Args:
        manifest: A loaded manifest.

    Returns:
        A dict of stage name to ``{'done': int, 'total': int, 'pending': [names]}``.
    """
    summary = {}
    for criterion, stage in (('pretrain', 'pretrain'), ('mortality', 'finetune')):
        trials = [
            t for t in manifest['trials']
            if criterion == 'pretrain' or t['needs_finetune']
        ]
        pending = [
            t['name'] for t in trials
            if not read_result(manifest, t, criterion).is_usable
        ]
        summary[stage] = {
            'done': len(trials) - len(pending),
            'total': len(trials),
            'pending': pending,
        }
    return summary
