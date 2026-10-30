#!/usr/bin/env python3
"""Render the pre-admission history sweep as a manuscript table.

Reads a `phase3_tuning.csv` written by `report_tuning_results.py` and emits the one table the
sweep supports: validation AUPRC against the cap on how many pre-admission records a model may
read. Reading the CSV rather than the evaluation YAMLs keeps this runnable away from the
cluster, where the per-trial `models/phase3_tuned_*` directories live.

Rows whose `hyperparameter` is not `HISTORY_LEN_STEPS` are dropped. That is what removes the
`<ablation:...>` rows an `EXTRA_TRIALS` entry contributes: an ablation is built from the arm's
defaults plus its own overrides, carries an empty `covers` list so it cannot enter a ranking,
and is not a cell of the grid. Reporting it as one would present a different model as a
history limit.

The sweep is one training run per arm on the tuning fold, so no arm has an interval and the
table cannot carry one. `--reference-sem` supplies the yardstick instead: the standard error of
a comparable model's test AUPRC across folds, from which the footnote derives the range six
independent draws would be expected to span. That expected range is what says whether the
observed spread is a result or noise.

Usage:
    python scripts/report_history_limit_sweep.py \\
        --csv 'tables/full value feature set experiments/phase3_tuning.csv' \\
        --output 'JMIR AI Round 2 Peer Review/history_limit_sweep.docx' \\
        --reference-sem 0.0068094 --reference-label 'peri-stay + history + text'
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.jmir.tables import Table, build_document, render_text


# Expected range of n independent standard normal draws, E[W_n]. Tabulated because the sweep
# has one arm count in practice and the closed form is an integral.
RANGE_CONSTANT = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847}


def read_arms(path: str, metric: str):
    """Read the sweep's arms from a phase3 tuning CSV.

    Args:
        path: Path to `phase3_tuning.csv`.
        metric: Column holding the selection metric value.

    Returns:
        List of (cap, value) sorted by cap descending.

    Raises:
        SystemExit: If the file holds no history arm, which means it is a different sweep.
    """
    arms = []
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('hyperparameter') != 'HISTORY_LEN_STEPS':
                continue
            if row.get('status') and row['status'] != 'ok':
                raise SystemExit(
                    f'arm {row.get("value")!r} has status {row["status"]!r}; the sweep did not '
                    f'finish and the table would be incomplete.'
                )
            arms.append((int(row['value']), float(row[metric])))
    if not arms:
        raise SystemExit(f'{path} holds no HISTORY_LEN_STEPS rows.')
    return sorted(arms, key=lambda item: -item[0])


def build_table(arms, number: str, precision: int, reference_sem=None,
                reference_label=None, reference_folds: int = 5) -> Table:
    """Assemble the table, with the noise yardstick as a footnote when one is supplied."""
    caps = [cap for cap, _ in arms]
    values = [value for _, value in arms]
    table = Table(
        number=number,
        caption=('Mortality validation set area under the precision-recall curve for models '
                 'trained with different limits on the number of pre-admission records '
                 'available as input'),
        stub_head='Maximum pre-admission records',
        columns=['AUPRC'],
    )

    cap_note = table.add_footnote(
        f'Each arm caps the number of most recent pre-admission records the model may read. '
        f'{max(caps)} is the extraction width, so that arm applies no cap; 0 withholds every '
        f'pre-admission record while keeping the peri-stay episode.'
    )
    table.stub_head = f'{table.stub_head}{cap_note}'

    if reference_sem:
        deviation = reference_sem * math.sqrt(reference_folds)
        constant = RANGE_CONSTANT.get(len(arms))
        observed = max(values) - min(values)
        expected = constant * deviation if constant else None
        text = (
            f'One training run per arm on the tuning fold, so no arm carries an interval. For '
            f'scale, the {reference_label} model reports a test AUPRC standard error of '
            f'{reference_sem:.4f} across {reference_folds} folds, a per-fold standard '
            f'deviation of {deviation:.4f}.'
        )
        if expected:
            text += (
                f' {len(arms)} independent draws at that variability would be expected to span '
                f'{expected:.4f}, against the {observed:.4f} observed here.'
            )
        note = table.add_footnote(text)
        table.columns = [f'AUPRC{note}']

    for cap, value in arms:
        table.add_row(str(cap), [f'{value:.{precision}f}'])
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Render the pre-admission history sweep as a manuscript table'
    )
    parser.add_argument('--csv', required=True, help='Path to phase3_tuning.csv')
    parser.add_argument('--output', required=True, help='Destination .docx')
    parser.add_argument('--metric', default='selection_value',
                        help='CSV column holding the metric (default: selection_value)')
    parser.add_argument('--table-number', default='1', help='Table number for the caption')
    parser.add_argument('--precision', type=int, default=4)
    parser.add_argument('--reference-sem', type=float, default=None,
                        help='Test-metric standard error of a comparable model, for the '
                             'footnote that scales the observed spread against noise')
    parser.add_argument('--reference-label', default='comparable',
                        help='How to name that model in the footnote')
    parser.add_argument('--reference-folds', type=int, default=5)
    parser.add_argument('--caption-style', default=None)
    args = parser.parse_args(argv)

    arms = read_arms(args.csv, args.metric)
    table = build_table(arms, args.table_number, args.precision, args.reference_sem,
                        args.reference_label, args.reference_folds)
    render_text(table)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    build_document(table, args.output, caption_style=args.caption_style)
    print(f'\nWrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
