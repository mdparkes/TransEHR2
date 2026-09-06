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

Text. The in-stay window closes at 48 h, before a discharge summary is written, so every text
record is pre-admission. A model reading in-stay records only therefore has no text available
to it, which is why experiments 10 and 15 carry none.
"""

import argparse
import os

import yaml


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
             'USE_INSTAY_RECORDS', 'COHORT_SUBSET', 'USE_HISTORICAL_RECORDS')

# (name, description, cohort, text, historical non-text, historical text, in-stay)
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
]


def build(base: dict, name: str, cohort: str, use_text: bool, historical_nontext: bool,
          historical_text: bool, instay: bool) -> dict:
    """Assemble one experiment config from the tuned base."""
    config = {key: value for key, value in base.items() if key not in DROP_KEYS}
    config['EXPERIMENT_NAME'] = name
    config['COHORT_SUBSET'] = cohort
    config['USE_TEXT'] = use_text
    config['USE_HISTORICAL_NONTEXT_RECORDS'] = historical_nontext
    config['USE_HISTORICAL_TEXT_RECORDS'] = historical_text
    config['USE_INSTAY_RECORDS'] = instay
    # With no history of either kind the region is dead weight, so crop it away rather than
    # masking 500 padded timesteps per episode. A run that keeps text must keep the region,
    # because all text is pre-admission.
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
