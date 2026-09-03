#!/usr/bin/env python3
"""Report diagnosis (phenotype) prediction results as a formatted table.

Reads the per-fold prediction CSVs written by
``dump_finetuned_predictions.py``, computes micro- and macro-averaged
multi-label classification metrics at decision thresholds calibrated on
a held-out split, compares every experiment against a nominated control
with the corrected resampled t test of Nadeau & Bengio (2003), controls
the false discovery rate with the Benjamini-Hochberg procedure, and
prints the resulting table to stdout and optionally to a Word document.

Because the task is multi-label, the threshold may be calibrated once
per label or once for all labels together. Per-label calibration lifts
the macro averages, since a single shared threshold is dominated by the
common labels and the rare ones rarely cross it; a shared threshold is
the more conservative choice for the rarest labels, whose calibration
split holds few positive examples.

Usage:
    python report_phenotype.py --experiments 3 9 1 2 7 --control 3 \
        --table-number 3 --output tables/table3_phenotype.docx

    # Shared threshold across all labels, for comparison
    python report_phenotype.py --experiments 3 9 1 2 7 --control 3 \
        --phenotype-threshold-scope global --table-number 3
"""

import sys

from reporting.cli import (F1_LABEL, CategorySpec, MetricSpec,
                                build_parser, describe_threshold, run)

TASK = 'phenotype'

DEFAULT_CAPTION = 'Diagnosis prediction evaluation results.'

# Accuracy is identical under micro and macro averaging because every
# label contributes the same number of predictions, so it is reported
# once above the two category headings.
SPECS = (
    MetricSpec('micro_accuracy', 'Accuracy'),
    CategorySpec('Microaverages'),
    MetricSpec('micro_f1', F1_LABEL, level=1),
    MetricSpec('micro_auroc', 'AUROC', level=1),
    MetricSpec('micro_auprc', 'AUPRC', level=1),
    MetricSpec('micro_recall_sensitivity', 'Sensitivity', level=1),
    MetricSpec('micro_specificity', 'Specificity', level=1),
    MetricSpec('micro_ppv', 'Positive predictive value', level=1),
    MetricSpec('micro_npv', 'Negative predictive value', level=1),
    MetricSpec('micro_false_positive_rate', 'False positive rate', level=1),
    MetricSpec('micro_false_negative_rate', 'False negative rate', level=1),
    MetricSpec('micro_false_discovery_rate', 'False discovery rate',
               level=1),
    MetricSpec('micro_prevalence', 'Prevalence', level=1, compare=False),
    CategorySpec('Macroaverages'),
    MetricSpec('macro_f1', F1_LABEL, level=1),
    MetricSpec('macro_auroc', 'AUROC', level=1),
    MetricSpec('macro_auprc', 'AUPRC', level=1),
    MetricSpec('macro_recall_sensitivity', 'Sensitivity', level=1),
    MetricSpec('macro_specificity', 'Specificity', level=1),
    MetricSpec('macro_ppv', 'Positive predictive value', level=1),
    MetricSpec('macro_npv', 'Negative predictive value', level=1),
    MetricSpec('macro_false_positive_rate', 'False positive rate', level=1),
    MetricSpec('macro_false_negative_rate', 'False negative rate', level=1),
    MetricSpec('macro_false_discovery_rate', 'False discovery rate',
               level=1),
    MetricSpec('macro_prevalence', 'Prevalence', level=1, compare=False),
)

# The published tables omit prevalence from the diagnosis tables, since
# it duplicates information already given for mortality.
DEFAULT_METRICS = tuple(
    spec.key for spec in SPECS
    if isinstance(spec, MetricSpec) and not spec.key.endswith('prevalence')
)


def main(argv=None):
    """Parse arguments and produce the diagnosis table.

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
        phenotype=True,
    )
    args = parser.parse_args(argv)
    if not args.metrics and not args.list_metrics:
        args.metrics = list(DEFAULT_METRICS)
    return run(args, TASK, SPECS, describe_threshold)


if __name__ == '__main__':
    sys.exit(main())
