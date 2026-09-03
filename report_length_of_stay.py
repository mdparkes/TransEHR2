#!/usr/bin/env python3
"""Report length-of-stay prediction results as a formatted table.

Reads the per-fold prediction CSVs written by
``dump_finetuned_predictions.py``, computes the regression metrics,
compares every experiment against a nominated control with the corrected
resampled t test of Nadeau & Bengio (2003), controls the false discovery
rate with the Benjamini-Hochberg procedure, and prints the resulting
table to stdout and optionally to a Word document.

Length of stay is a regression task, so there is no decision threshold
to set.

Usage:
    python report_length_of_stay.py --experiments 3 9 1 2 7 --control 3 \
        --table-number 2 --output tables/table2_length_of_stay.docx
"""

import sys

from reporting.cli import MetricSpec, build_parser, run

TASK = 'length_of_stay'

DEFAULT_CAPTION = 'Length-of-stay evaluation results.'

SPECS = (
    MetricSpec('mean_absolute_error', 'Mean absolute error, hours'),
    MetricSpec('concordance_index', 'Concordance index'),
)


def main(argv=None):
    """Parse arguments and produce the length-of-stay table.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = build_parser(
        TASK,
        description=__doc__.split('\n\n')[0],
        default_caption=DEFAULT_CAPTION,
        classification=False,
    )
    args = parser.parse_args(argv)
    return run(args, TASK, SPECS)


if __name__ == '__main__':
    sys.exit(main())
