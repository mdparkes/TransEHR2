"""What each prediction task reports, and how its table is configured.

The three tasks differ in their metrics, in whether a decision threshold applies, and in
whether they are multi-label. Holding that here keeps `report_results_tables.py` about the
experiment design -- which models are compared, against which control, in what order -- and
keeps `reporting.cli` about tables and statistics.
"""

from reporting.cli import CategorySpec, F1_LABEL, MetricSpec, describe_threshold


TASKS = ('mortality', 'length_of_stay', 'phenotype')


class TaskSpec:
    """How one task's table is built.

    Attributes:
        key: Task name, as it appears in the prediction CSV paths.
        caption: Caption stem; the cohort is appended by the caller.
        specs: Metric and category rows, in table order.
        classification: Whether the task has a decision threshold to calibrate.
        phenotype: Whether the task is multi-label, which adds the threshold-scope option.
        threshold_note: Builder for the threshold footnote, or None where none applies.
        default_metrics: Rows to report when --metrics is not given, or None for all of them.
    """

    def __init__(self, key, caption, specs, classification, phenotype=False,
                 threshold_note=None, default_metrics=None):
        self.key = key
        self.caption = caption
        self.specs = specs
        self.classification = classification
        self.phenotype = phenotype
        self.threshold_note = threshold_note
        self.default_metrics = default_metrics


MORTALITY_SPECS = (
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


LENGTH_OF_STAY_SPECS = (
    MetricSpec('mean_absolute_error', 'Mean absolute error, hours'),
    MetricSpec('concordance_index', 'Concordance index'),
)


# Accuracy is identical under micro and macro averaging because every
# label contributes the same number of predictions, so it is reported
# once above the two category headings.
PHENOTYPE_SPECS = (
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
    spec.key for spec in PHENOTYPE_SPECS
    if isinstance(spec, MetricSpec) and not spec.key.endswith('prevalence')
)


TASK_SPECS = {
    'mortality': TaskSpec(
        'mortality', 'In-hospital mortality', MORTALITY_SPECS,
        classification=True, threshold_note=describe_threshold,
    ),
    'length_of_stay': TaskSpec(
        # Regression, so there is no decision threshold to calibrate.
        'length_of_stay', 'Length of stay', LENGTH_OF_STAY_SPECS,
        classification=False,
    ),
    'phenotype': TaskSpec(
        'phenotype', 'Diagnosis prediction', PHENOTYPE_SPECS,
        classification=True, phenotype=True, threshold_note=describe_threshold,
        default_metrics=DEFAULT_METRICS,
    ),
}
