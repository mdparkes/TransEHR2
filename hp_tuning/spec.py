"""Expansion of a hyperparameter tuning spec into one experiment config per trial.

The revision plan's Phase 2 is an *additive* (coordinate) sweep: every tested value is paired
against every other hyperparameter's default, so the sweep is a shared all-defaults centre plus
one trial per non-default value. Five hyperparameters at three values each is therefore eleven
configurations per encoding arm, not 3^5.

Expanding that into standalone config files, rather than looping inside one long-lived process,
is what makes the sweep fan out. Each trial is an independent single-GPU job with its own
config, its own model directory and its own evaluation YAML, so the node fills to whatever
depth the scheduler allows, a failed trial is re-runnable on its own, and nothing in the sweep
touches distributed code.

The manifest this module writes is the contract between the three stages:
``generate_tuning_configs.py`` writes it, the SLURM array scripts index into its ``trials``
list, and ``report_tuning_results.py`` and ``select_tuned_hyperparameters.py`` read results
back through it.
"""

import itertools
import os
import textwrap

import yaml

from typing import Any, Dict, List, Optional


# The two ways a hyperparameter can be judged, and what each one reads.
#
# Which applies is set by what the hyperparameter does to the objective. Learning rate and its
# decay leave the pretraining loss intact, so that loss ranks them directly and no finetune is
# needed. The masking ratios and the time weight *rescale* the objective -- a model masked at
# 0.75 is not solving the same problem as one masked at 0.25 -- so their pretraining losses are
# not comparable to each other and selection has to move downstream to a task metric.
# How a grid is turned into trials.
#   'additive'  -- each untested value paired against every other hyperparameter's default,
#                  so the sweep is one shared centre plus one trial per non-default value.
#                  Each hyperparameter is then ranked on its own coordinate.
#   'factorial' -- every combination of every value. Ranking a single hyperparameter over a
#                  cross product has no unambiguous meaning, because each of its values
#                  appears in several cells, so selection ranks whole cells instead. Use it
#                  where the hyperparameters interact and the best pair is not the pair of
#                  individual bests.
DESIGNS = ('additive', 'factorial')

SELECTION_CRITERIA = {
    'pretrain': {
        'task': 'pretrain',
        'evaluation_file': 'evaluation_pretrained.yaml',
        'block': 'val_losses',
        'metric': 'Optimization_Loss',
        'direction': 'min',
        'needs_finetune': False,
    },
    'mortality': {
        'task': 'mortality',
        'evaluation_file': 'evaluation_mortality.yaml',
        'block': 'validation_scores',
        'metric': 'AUPRC',
        'direction': 'max',
        'needs_finetune': True,
    },
}


def slugify_value(value: Any) -> str:
    """Render a hyperparameter value as a filesystem- and path-safe token.

    Experiment names become directory names under ``models/``, ``log/`` and ``checkpoints/``,
    so the token has to survive a path without quoting and has to be unique across the grid.
    ``%g`` keeps 2e-05 from becoming 0p00002, and the sign and point substitutions keep the
    result free of characters that a shell would want to interpret.

    Args:
        value: The value to render.

    Returns:
        A token containing only letters, digits and underscores.
    """
    if value is None:
        return 'none'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        text = f'{value:g}'
    else:
        text = str(value)
    return (text.replace('-', 'm').replace('+', '')
                .replace('.', 'p').replace(' ', '_'))


def load_spec(spec_path: str) -> Dict[str, Any]:
    """Read a tuning spec and check it hard enough to fail before any GPU time is spent.

    Args:
        spec_path: Path to the spec YAML.

    Returns:
        The spec as a dict, with relative paths resolved against the spec's own directory so
        that a spec can be run from anywhere.

    Raises:
        ValueError: If a required key is missing, a grid is malformed, or a selection
            criterion is not one this module knows how to read back.
    """
    with open(spec_path, 'r') as f_in:
        spec = yaml.safe_load(f_in)

    for key in ('SPEC_NAME', 'BASE_CONFIG', 'DATASET_CONFIG', 'OUTPUT_DIR', 'MANIFEST',
                'ARMS', 'GRID', 'FOLD'):
        if key not in spec:
            raise ValueError(f"{spec_path} is missing the required key {key!r}")

    if not spec['ARMS']:
        raise ValueError(f"{spec_path} defines no arms")
    if not spec['GRID']:
        raise ValueError(f"{spec_path} defines no grid")

    spec['DESIGN'] = spec.get('DESIGN', 'additive')
    if spec['DESIGN'] not in DESIGNS:
        raise ValueError(
            f"{spec_path} sets DESIGN to {spec['DESIGN']!r}; expected one of {list(DESIGNS)}"
        )

    for hp_name, entry in spec['GRID'].items():
        if 'values' not in entry or not entry['values']:
            raise ValueError(f"{spec_path}: grid entry {hp_name!r} has no values")
        criterion = entry.get('select_on')
        if criterion not in SELECTION_CRITERIA:
            raise ValueError(
                f"{spec_path}: grid entry {hp_name!r} selects on {criterion!r}; "
                f"expected one of {sorted(SELECTION_CRITERIA)}"
            )
        values = entry['values']
        if len(set(map(repr, values))) != len(values):
            raise ValueError(
                f"{spec_path}: grid entry {hp_name!r} repeats a value: {values}. The first "
                f"entry is the default and is run once as the shared centre, so a repeat "
                f"would silently collapse two trials into one."
            )

    if spec['DESIGN'] == 'factorial':
        criteria = sorted({entry['select_on'] for entry in spec['GRID'].values()})
        if len(criteria) > 1:
            raise ValueError(
                f"{spec_path} is factorial but its grid selects on {criteria}. A factorial is "
                f"ranked as whole cells against one criterion, so every hyperparameter in it "
                f"has to share one. Split the grid into one sweep per criterion."
            )

    # Extra trials are one-off ablations that hang off the sweep rather than sitting in the
    # grid: a single named change from the centre, measured head-to-head against it. They are
    # deliberately NOT grid cells -- an ablation has no ordered set of values to rank and must
    # never influence which grid value is selected.
    for index, extra in enumerate(spec.get('EXTRA_TRIALS') or []):
        where = f"{spec_path}: EXTRA_TRIALS[{index}]"
        for key in ('name', 'arm', 'select_on', 'overrides'):
            if key not in extra:
                raise ValueError(f"{where} is missing the required key {key!r}")
        if extra['arm'] not in spec['ARMS']:
            raise ValueError(
                f"{where} names arm {extra['arm']!r}, which is not in ARMS "
                f"({sorted(spec['ARMS'])})"
            )
        if extra['select_on'] not in SELECTION_CRITERIA:
            raise ValueError(
                f"{where} selects on {extra['select_on']!r}; "
                f"expected one of {sorted(SELECTION_CRITERIA)}"
            )
        if not extra['overrides']:
            raise ValueError(
                f"{where} overrides nothing, so it would duplicate the centre it is meant to "
                f"be compared against."
            )
        collides = sorted(set(extra['overrides']) & set(spec['GRID']))
        if collides:
            raise ValueError(
                f"{where} overrides {collides}, which the grid also sweeps. An extra trial is "
                f"read as one change from the centre; changing a swept hyperparameter as well "
                f"would confound it with that hyperparameter's own coordinate."
            )

    spec_dir = os.path.dirname(os.path.abspath(spec_path))
    for key in ('BASE_CONFIG', 'DATASET_CONFIG', 'OUTPUT_DIR', 'MANIFEST'):
        if not os.path.isabs(spec[key]):
            spec[key] = os.path.normpath(os.path.join(spec_dir, spec[key]))
    spec['SPEC_PATH'] = os.path.abspath(spec_path)

    # MODEL_DIR is where the trials will write their evaluation YAMLs, so the reporting stage
    # needs it. It belongs to the base config rather than the spec -- one source of truth, and
    # a spec that disagreed with the configs it generates would send the reporter looking in
    # the wrong tree.
    if not os.path.exists(spec['BASE_CONFIG']):
        raise ValueError(
            f"{spec_path} points at a base config that does not exist:\n"
            f"    {spec['BASE_CONFIG']}\n"
            f"If this sweep runs on the settings a previous phase selected, that config is "
            f"written by select_tuned_hyperparameters.py and the previous phase has to finish "
            f"first."
        )
    with open(spec['BASE_CONFIG'], 'r') as f_in:
        base_config = yaml.safe_load(f_in)
    if 'MODEL_DIR' not in base_config:
        raise ValueError(f"{spec['BASE_CONFIG']} does not define MODEL_DIR")
    spec['MODEL_DIR'] = base_config['MODEL_DIR']

    return spec


def _extra_trials(
    spec: Dict[str, Any],
    arm_name: str,
    arm_overrides: Dict[str, Any],
    defaults: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """One-off ablations hanging off one arm of a sweep.

    Args:
        spec: A spec as returned by :func:`load_spec`.
        arm_name: The arm these belong to.
        arm_overrides: The config keys that define that arm.
        defaults: The default value of every grid hyperparameter.

    Returns:
        Trial dicts, in spec order.
    """
    trials = []
    for extra in spec.get('EXTRA_TRIALS') or []:
        if extra['arm'] != arm_name:
            continue
        criterion = SELECTION_CRITERIA[extra['select_on']]
        # Built from the DEFAULTS, not from any grid cell, so the only thing separating it
        # from the arm's centre is its own overrides. That is what makes the head-to-head
        # against the centre a one-variable comparison.
        overrides = dict(arm_overrides, **defaults)
        overrides.update(extra['overrides'])
        trials.append({
            'name': f"{spec['SPEC_NAME']}_{arm_name}_{extra['name']}",
            'arm': arm_name,
            'is_centre': False,
            'hyperparameter': None,
            'value': None,
            'overrides': overrides,
            'cell': None,
            # Empty on purpose: `covers` is how the ranking stage finds the trial for a
            # grid cell, so an empty list keeps an ablation out of every ranking and out
            # of selection. An extra trial is reported, never selected on.
            'covers': [],
            'needs_finetune': criterion['needs_finetune'],
            'is_extra': True,
            'select_on': extra['select_on'],
            'description': extra.get('description', ''),
        })
    return trials


def _factorial_cells(
    spec: Dict[str, Any],
    arm_name: str,
    arm_overrides: Dict[str, Any],
    defaults: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Every combination of every grid value, for one arm.

    A cell is not the trial for any single grid value: each value appears in as many cells as
    the other hyperparameters have combinations, so a per-value lookup would be ambiguous.
    ``covers`` is therefore empty and ``cell`` carries the full assignment, which is what the
    ranking stage reads. The all-defaults cell is still flagged as the centre, so the arm
    comparison keeps a defined pair of runs to sit on.

    Args:
        spec: A spec as returned by :func:`load_spec`.
        arm_name: The arm these belong to.
        arm_overrides: The config keys that define that arm.
        defaults: The default value of every grid hyperparameter.

    Returns:
        Trial dicts, one per cell, in grid order with the first value of each varying slowest.
    """
    grid = spec['GRID']
    aliases = spec.get('ALIASES', {})
    hp_names = list(grid.keys())
    needs_finetune = SELECTION_CRITERIA[grid[hp_names[0]]['select_on']]['needs_finetune']

    trials = []
    for combination in itertools.product(*(grid[name]['values'] for name in hp_names)):
        cell = dict(zip(hp_names, combination))
        token = '_'.join(
            f"{aliases.get(name, name.lower())}_{slugify_value(cell[name])}"
            for name in hp_names
        )
        trials.append({
            'name': f"{spec['SPEC_NAME']}_{arm_name}_{token}",
            'arm': arm_name,
            'is_centre': all(cell[name] == defaults[name] for name in hp_names),
            'hyperparameter': None,
            'value': None,
            'overrides': dict(arm_overrides, **cell),
            'cell': cell,
            'covers': [],
            'needs_finetune': needs_finetune,
            'is_extra': False,
            'select_on': grid[hp_names[0]]['select_on'],
            'description': '',
        })
    return trials


def expand_trials(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn a spec into the list of trials the sweep consists of.

    One centre per arm carries every hyperparameter at its default, which is why the sweep is
    ``1 + sum(len(values) - 1)`` trials per arm rather than one per grid cell. The centre is
    recorded as *covering* the default value of every hyperparameter, so the reporting stage
    can look up "the trial that ran PRETRAIN_LEARNING_RATE at 0.002" without special-casing it.

    Args:
        spec: A spec as returned by :func:`load_spec`.

    Returns:
        Trial dicts, centres first within each arm, in a stable order. The order is the array
        index the SLURM jobs use, so it must not depend on dict iteration luck.
    """
    grid = spec['GRID']
    aliases = spec.get('ALIASES', {})
    hp_names = list(grid.keys())
    defaults = {name: grid[name]['values'][0] for name in hp_names}

    trials = []
    for arm_name in spec['ARMS']:
        arm_overrides = spec['ARMS'][arm_name] or {}

        if spec.get('DESIGN', 'additive') == 'factorial':
            trials.extend(_factorial_cells(spec, arm_name, arm_overrides, defaults))
            trials.extend(_extra_trials(spec, arm_name, arm_overrides, defaults))
            continue

        centre_name = f"{spec['SPEC_NAME']}_{arm_name}_centre"
        trials.append({
            'name': centre_name,
            'arm': arm_name,
            'is_centre': True,
            'hyperparameter': None,
            'value': None,
            'overrides': dict(arm_overrides, **defaults),
            'cell': None,
            # The centre is the default trial for every hyperparameter at once.
            'covers': [{'hyperparameter': name, 'value': defaults[name]} for name in hp_names],
            'is_extra': False,
            # Every hyperparameter that selects downstream needs the centre as its baseline,
            # so the centre is finetuned whenever any of them do.
            'needs_finetune': any(
                SELECTION_CRITERIA[grid[name]['select_on']]['needs_finetune']
                for name in hp_names
            ),
        })

        for hp_name in hp_names:
            alias = aliases.get(hp_name, hp_name.lower())
            criterion = SELECTION_CRITERIA[grid[hp_name]['select_on']]
            for value in grid[hp_name]['values'][1:]:
                overrides = dict(arm_overrides, **defaults)
                overrides[hp_name] = value
                trials.append({
                    'name': f"{spec['SPEC_NAME']}_{arm_name}_{alias}_{slugify_value(value)}",
                    'arm': arm_name,
                    'is_centre': False,
                    'hyperparameter': hp_name,
                    'value': value,
                    'overrides': overrides,
                    'cell': None,
                    'covers': [{'hyperparameter': hp_name, 'value': value}],
            'is_extra': False,
                    'needs_finetune': criterion['needs_finetune'],
                })

        trials.extend(_extra_trials(spec, arm_name, arm_overrides, defaults))

    names = [trial['name'] for trial in trials]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            f"Trial names collide: {duplicates}. Two grid values slugified to the same token, "
            f"which would make two trials share a model directory and overwrite each other. "
            f"Give them distinguishable values or set an ALIASES entry."
        )
    return trials


def write_trial_configs(
        spec: Dict[str, Any],
        trials: List[Dict[str, Any]],
        overwrite: bool = False
) -> List[Dict[str, Any]]:
    """Write one experiment config per trial and return the trials with their paths attached.

    Args:
        spec: A spec as returned by :func:`load_spec`.
        trials: Trials as returned by :func:`expand_trials`.
        overwrite: Replace configs that already exist. Off by default: a config file is the
            record of what a finished run actually ran, and rewriting one silently
            re-describes results that are already on disk.

    Returns:
        The trials, each with a ``config`` key holding the path written.

    Raises:
        FileExistsError: If a config exists and ``overwrite`` is False.
        ValueError: If an override names a key the base config does not define.
    """
    with open(spec['BASE_CONFIG'], 'r') as f_in:
        base_config = yaml.safe_load(f_in)

    if 'HYPERPARAMETERS_TO_TUNE' in base_config:
        raise ValueError(
            f"{spec['BASE_CONFIG']} carries HYPERPARAMETERS_TO_TUNE. The base config is a "
            f"single complete configuration; the grid lives in the spec."
        )

    os.makedirs(spec['OUTPUT_DIR'], exist_ok=True)

    for trial in trials:
        unknown = [key for key in trial['overrides'] if key not in base_config]
        if unknown:
            raise ValueError(
                f"Trial {trial['name']} overrides {unknown}, which the base config "
                f"{spec['BASE_CONFIG']} does not define. A key that is absent from the base "
                f"is almost always a typo -- the run would take the code default instead and "
                f"the sweep would silently test nothing."
            )

        config = dict(base_config)
        config.update(trial['overrides'])
        config['EXPERIMENT_NAME'] = trial['name']

        config_path = os.path.join(spec['OUTPUT_DIR'], f"{trial['name']}.yaml")
        if os.path.exists(config_path) and not overwrite:
            raise FileExistsError(
                f"{config_path} already exists. Pass --overwrite to replace it, but check "
                f"first whether results under models/{trial['name']}/ were produced by the "
                f"version on disk."
            )

        if trial['is_centre']:
            described = 'the shared all-defaults centre'
        elif trial.get('is_extra', False):
            changed = ', '.join(
                f'{key} = {value!r}' for key, value in sorted(trial['overrides'].items())
                if key not in (spec['GRID'].keys() | set(spec['ARMS'][trial['arm']] or {}))
                and base_config.get(key) != value
            )
            described = f"ablation {trial['name'].rsplit('_', 1)[-1]} -- {changed}"
        else:
            described = f"{trial['hyperparameter']} = {trial['value']!r}"

        if trial.get('is_extra', False):
            rationale = (
                "# This is an ablation, not a grid cell. It sits at the centre in every\n"
                "# respect but the setting above, so it is read head-to-head against the\n"
                "# centre, and it takes no part in selecting any hyperparameter's value.\n"
            )
            if trial.get('description'):
                wrapped = '\n'.join(
                    f'#   {line}' for line in
                    textwrap.wrap(' '.join(trial['description'].split()), width= 88)
                )
                rationale += f'#\n{wrapped}\n'
        else:
            rationale = (
                "# Every other tuned hyperparameter sits at its default, which is what makes\n"
                "# this an additive sweep: each value is measured against one shared point.\n"
            )

        header = (
            f"# Generated by generate_tuning_configs.py from {spec['SPEC_PATH']}\n"
            f"# Do not edit by hand -- regenerate the spec instead.\n"
            f"#\n"
            f"# Sweep: {spec['SPEC_NAME']}   Arm: {trial['arm']}   Trial: {described}\n"
            f"{rationale}"
            f"\n"
        )
        with open(config_path, 'w') as f_out:
            f_out.write(header)
            yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)

        trial['config'] = config_path

    return trials


def write_manifest(spec: Dict[str, Any], trials: List[Dict[str, Any]]) -> str:
    """Write the manifest that the SLURM arrays and the reporting scripts read.

    Args:
        spec: A spec as returned by :func:`load_spec`.
        trials: Trials with their ``config`` paths attached.

    Returns:
        The manifest path written.
    """
    grid = spec['GRID']
    manifest = {
        'spec_name': spec['SPEC_NAME'],
        'spec_path': spec['SPEC_PATH'],
        'base_config': spec['BASE_CONFIG'],
        'config_dir': spec['OUTPUT_DIR'],
        'dataset_config': spec['DATASET_CONFIG'],
        'model_dir': spec['MODEL_DIR'],
        'fold': spec['FOLD'],
        'arms': list(spec['ARMS'].keys()),
        'design': spec.get('DESIGN', 'additive'),
        'grid': {
            name: {
                'values': entry['values'],
                'default': entry['values'][0],
                'select_on': entry['select_on'],
            }
            for name, entry in grid.items()
        },
        'selection_criteria': SELECTION_CRITERIA,
        'trials': [
            {
                'name': trial['name'],
                'config': trial['config'],
                'arm': trial['arm'],
                'is_centre': trial['is_centre'],
                'hyperparameter': trial['hyperparameter'],
                'value': trial['value'],
                'cell': trial.get('cell'),
                'covers': trial['covers'],
                'needs_finetune': trial['needs_finetune'],
                'is_extra': trial.get('is_extra', False),
                'select_on': trial.get('select_on'),
                'description': trial.get('description', ''),
            }
            for trial in trials
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(spec['MANIFEST'])), exist_ok=True)
    with open(spec['MANIFEST'], 'w') as f_out:
        yaml.dump(manifest, f_out, default_flow_style=False, sort_keys=False)
    return spec['MANIFEST']


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Read a manifest written by :func:`write_manifest`.

    Args:
        manifest_path: Path to the manifest YAML.

    Returns:
        The manifest as a dict.
    """
    with open(manifest_path, 'r') as f_in:
        return yaml.safe_load(f_in)


def pretrain_trials(manifest: Dict[str, Any], arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """The trials that need a pretraining job: all of them, optionally filtered by arm.

    Args:
        manifest: A loaded manifest.
        arm: Restrict to one encoding arm, or None for every arm.

    Returns:
        The matching trials, in manifest order.
    """
    return [t for t in manifest['trials'] if arm is None or t['arm'] == arm]


def finetune_trials(manifest: Dict[str, Any], arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """The trials that need a finetuning job.

    These are the centre of each arm plus every trial whose hyperparameter is selected on a
    downstream task -- seven per arm under the Phase 2 grid, against eleven pretrains.

    Args:
        manifest: A loaded manifest.
        arm: Restrict to one encoding arm, or None for every arm.

    Returns:
        The matching trials, in manifest order.
    """
    return [
        t for t in manifest['trials']
        if t['needs_finetune'] and (arm is None or t['arm'] == arm)
    ]


def extra_trials(manifest: Dict[str, Any], arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """The one-off ablations attached to a sweep, as distinct from its grid cells.

    These are reported and never selected on: each is a single named change from its arm's
    centre, so it reads as a head-to-head rather than as a coordinate with an ordered set of
    values to rank.

    Args:
        manifest: A loaded manifest.
        arm: Restrict to one encoding arm, or None for every arm.

    Returns:
        The matching trials, in manifest order. Empty for a sweep with no EXTRA_TRIALS block,
        including any manifest written before extras existed.
    """
    return [
        t for t in manifest['trials']
        if t.get('is_extra', False) and (arm is None or t['arm'] == arm)
    ]
