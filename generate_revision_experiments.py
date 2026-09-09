#!/usr/bin/env python3
"""Write the revision's experiment configs from the tuned configuration.

Every experiment inherits the hyperparameters the tuning phases settled and differs only in
which records reach the model and which patients it runs on. Keeping that difference in one
table, rather than in eight near-identical YAML files, is what makes the design reviewable: the
table below is the experiment matrix.

The generated configs match the ignored `experiment*.yaml` pattern, so the repository holds the
generator rather than its output and cannot end up carrying two configurations that claim to
describe the same experiment.

Usage:
    python generate_revision_experiments.py
    python generate_revision_experiments.py --dry_run

Cohorts. Comparing a model that reads pre-admission history against one that does not is
diluted by episodes with no history to read, and the paired tests need both arms on the same
episodes. `discharge_summary` keeps episodes with at least one pre-admission discharge summary;
`any_history` keeps those with at least one pre-admission value-stream record, which is the
wider cohort for contrasts the narrower one underpowers.

Experiment 18 takes its cohort as an explicit episode manifest instead, through
COHORT_EPISODES. It is the in-stay-only control that `run_charlson_logistic_regression.py` is
compared against, and that comparison's cohort is the episodes for which a Charlson index, an
age and a sex all exist -- which is not a predicate over the extracted arrays. Handing both
arms the one file `compute_charlson_index.py --write_cohort` writes is what puts them on the
same episodes. Generate the configs after writing it: the path is recorded here, and
`run_experiment.py` refuses to start if it is missing.

Text. Only pre-admission text reaches the model: a text feature is a discharge-time artifact
of an admission, so `collate_tensorized` drops any text record at or after admission rather
than let a stay's own discharge documentation predict its outcome. A model reading in-stay
records only therefore has no text available to it, which is why experiments 10 and 15 carry
none.
"""

import argparse
import os

import yaml

from TransEHR2.data.cohorts import COHORTS


REPO = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments', 'tuning',
                           'phase3_base.yaml')
OUTPUT_DIR = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments')

# Keys this generator sets itself. Everything else in the base config carries through, the
# seeds included: all eight experiments share one seed pair, so a contrast between them is
# paired on initialisation and batch order as well as on fold and episode, and each run
# reproduces.
DROP_KEYS = ('EXPERIMENT_NAME', 'HISTORY_LEN_STEPS', 'USE_TEXT',
             'USE_HISTORICAL_NONTEXT_RECORDS', 'USE_HISTORICAL_TEXT_RECORDS',
             'USE_INSTAY_RECORDS', 'COHORT_SUBSET', 'COHORT_EPISODES',
             'USE_HISTORICAL_RECORDS')

# Written by compute_charlson_index.py --write_cohort; see the note on cohorts above.
CHARLSON_COHORT = os.path.join('misc', 'charlson', 'charlson_cohort.txt')

# (name, description, cohort, text, historical non-text, historical text, in-stay). A cohort
# that is not one of the named predicates is taken as a path to an episode manifest.
EXPERIMENTS = [
    ('experiment10_instay_dischargesubset_rev',
     'In-Stay Records Only, Patients With At Least 1 Discharge Summary',
     'discharge_summary', False, False, False, True),
    ('experiment11_history_text_dischargesubset_rev',
     'Historical Records Only, Text Features, Patients With At Least 1 Discharge Summary',
     'discharge_summary', True, True, True, False),
    ('experiment12_history_instay_notext_dischargesubset_rev',
     'In-Stay + Historical Records, No Text Features, '
     'Patients With At Least 1 Discharge Summary',
     'discharge_summary', False, True, False, True),
    ('experiment13_history_instay_text_dischargesubset_rev',
     'In-Stay + Historical Records, Text Features, '
     'Patients With At Least 1 Discharge Summary',
     'discharge_summary', True, True, True, True),
    ('experiment14_instay_textonly_dischargesubset_rev',
     'In-Stay + Text Features Only, Patients With At Least 1 Discharge Summary',
     'discharge_summary', True, False, True, True),
    ('experiment15_instay_historysubset_rev',
     'In-Stay Records Only, Patients With At Least 1 Historical Record',
     'any_history', False, False, False, True),
    ('experiment16_history_text_historysubset_rev',
     'Historical Records Only, Text Features, Patients With At Least 1 Historical Record',
     'any_history', True, True, True, False),
    ('experiment17_history_instay_text_historysubset_rev',
     'In-Stay + Historical Records, Text Features, '
     'Patients With At Least 1 Historical Record',
     'any_history', True, True, True, True),
    ('experiment18_instay_charlsonsubset_rev',
     'In-Stay Records Only, Patients With A Charlson Comorbidity Index',
     CHARLSON_COHORT, False, False, False, True),
]


def build(base: dict, name: str, cohort: str, use_text: bool, historical_nontext: bool,
          historical_text: bool, instay: bool) -> dict:
    """Assemble one experiment config from the tuned base."""
    config = {key: value for key, value in base.items() if key not in DROP_KEYS}
    config['EXPERIMENT_NAME'] = name
    # A named predicate goes in COHORT_SUBSET; anything else is a path to an episode manifest.
    if cohort in COHORTS:
        config['COHORT_SUBSET'] = cohort
    else:
        config['COHORT_EPISODES'] = cohort
    config['USE_TEXT'] = use_text
    config['USE_HISTORICAL_NONTEXT_RECORDS'] = historical_nontext
    config['USE_HISTORICAL_TEXT_RECORDS'] = historical_text
    config['USE_INSTAY_RECORDS'] = instay
    # With no history of either kind the region is dead weight, so crop it away rather than
    # masking 500 padded timesteps per episode. A run that keeps text must keep the region:
    # only pre-admission text reaches the model, so cropping the region removes all of it.
    config['HISTORY_LEN_STEPS'] = 0 if not (historical_nontext or historical_text) else None
    return config


def header(name: str, description: str) -> str:
    return (f'# {description}\n'
            f'#\n'
            f'# Written by generate_revision_experiments.py from the tuned configuration in\n'
            f'# tuning/phase3_base.yaml. Edit the generator, not this file.\n\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    parser.add_argument('--base_config', default=BASE_CONFIG)
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the matrix without writing anything')
    args = parser.parse_args(argv)

    with open(args.base_config) as handle:
        base = yaml.safe_load(handle)

    width = max(len(name) for name, *_ in EXPERIMENTS)
    print(f"{'experiment':{width}}  {'cohort':17}  {'text':>5}  {'h-nontext':>9}  "
          f"{'h-text':>6}  {'in-stay':>7}  {'hist steps':>10}")
    print('-' * (width + 66))
    for name, description, cohort, use_text, nontext, text, instay in EXPERIMENTS:
        config = build(base, name, cohort, use_text, nontext, text, instay)
        steps = config['HISTORY_LEN_STEPS']
        print(f'{name:{width}}  {cohort:17}  {str(use_text):>5}  {str(nontext):>9}  '
              f'{str(text):>6}  {str(instay):>7}  '
              f"{'all' if steps is None else steps:>10}")
        if args.dry_run:
            continue
        path = os.path.join(args.output_dir, f'{name}.yaml')
        with open(path, 'w') as handle:
            handle.write(header(name, description))
            yaml.dump(config, handle, sort_keys=True, default_flow_style=False)

    if args.dry_run:
        print('\nDry run: nothing written.')
    else:
        print(f'\nWrote {len(EXPERIMENTS)} configs to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
