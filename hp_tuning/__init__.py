"""Machinery for the Phase 2 additive hyperparameter sweep.

Three stages, three command-line entry points at the repository root:

    generate_tuning_configs.py      spec -> one experiment config per trial, plus a manifest
    report_tuning_results.py        manifest -> progress, tables, and the ranking of each value
    select_tuned_hyperparameters.py manifest -> the winning value of each hyperparameter, and
                                    an assembled config to carry into the next phase

The trials themselves are ordinary single-GPU ``run_experiment.py`` jobs. Nothing in this
package trains anything; it decides what to run and reads back what ran.
"""
