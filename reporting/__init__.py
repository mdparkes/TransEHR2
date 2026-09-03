"""Evaluation, statistics and table building for TransEHR2 results.

Loads the per-fold predictions, computes metrics, compares models with the
corrected resampled t test of Nadeau & Bengio (2003), and assembles the
result into a table. Publisher house style -- number formatting and table
layout -- lives in :mod:`reporting.jmir` and is the only part that has to
change to target a different one.
"""
