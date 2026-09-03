#!/usr/bin/env python3
"""Report in-hospital mortality prediction results as a formatted table.

Reads the per-fold prediction CSVs written by
``dump_finetuned_predictions.py``, computes the binary classification
metrics at a decision threshold that is calibrated on a held-out split,
compares every experiment against a nominated control with the corrected
resampled t test of Nadeau & Bengio (2003), controls the false discovery
rate with the Benjamini-Hochberg procedure, and prints the resulting
table to stdout and optionally to a Word document.

Usage:
    python report_mortality.py --experiments 3 9 1 2 7 --control 3 \
        --table-number 1 --output tables/table1_mortality.docx

    # Fixed-threshold sensitivity analysis
    python report_mortality.py --experiments 3 9 1 2 7 --control 3 \
        --threshold 0.5 --table-number 1
"""

import sys

from reporting.cli import (F1_LABEL, MetricSpec, build_parser,
                                describe_threshold, run)

TASK = 'mortality'

DEFAULT_CAPTION = 'In-hospital mortality evaluation results.'

SPECS = (
    MetricSpec('accuracy', 'Accuracy'),
    MetricSpec('f1', F1_LABEL),
    MetricSpec('auroc', 'AUROC'),
    MetricSpec('auprc', 'AUPRC'),
    MetricSpec('recall_sensitivity', 'Sensitivity'),
    MetricSpec('specificity', 'Specificity'),
    MetricSpec('ppv', 'Positive predictive value'),
    MetricSpec('npv', 'Negative predictive value'),
    MetricSpec('false_positive_rate', 'False positive rate'),
    MetricSpec('false_negative_rate', 'False negative rate'),
    MetricSpec('false_discovery_rate', 'False discovery rate'),
    MetricSpec('prevalence', 'Prevalence', compare=False),
)


def main(argv=None):
    """Parse arguments and produce the mortality table.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = build_parser(
        TASK,
        description=__doc__.split('\n\n')[0],
        default_caption=DEFAULT_CAPTION,
        classification=True,
    )
    args = parser.parse_args(argv)
    return run(args, TASK, SPECS, describe_threshold)


if __name__ == '__main__':
    sys.exit(main())
