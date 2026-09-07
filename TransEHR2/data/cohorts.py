"""Cohort predicates over the extracted arrays.

A cohort restricts the experiments to episodes that carry a particular kind of pre-admission
record. The comparison between a model that reads history and one that does not is otherwise
biased by the episodes that have no history to read: those contribute identically to both arms
and dilute the contrast, and the paired tests need the two arms to run on the same episodes.

The predicates read the extracted arrays rather than the source CSVs, so they select within an
extraction that already exists and cost no re-extraction. Cohort membership is computed on the
full extracted history and does not depend on the runtime `HISTORY_LEN_STEPS` crop -- an
episode does not leave the cohort because a sweep arm stopped showing the model its records.

Selection is per episode, matching `filter_listfiles_by_discharge_summary.py`. An episode with
no pre-admission record is what biases the comparison, whether or not the same patient has
another episode that does.
"""

from typing import Optional

import numpy as np


# Position of the discharge summary in the dataset config's TEXT_FEATS list.
DISCHARGE_SUMMARY_INDEX = 0

COHORTS = ('discharge_summary', 'any_history')


def history_observed(masks, max_history_len_steps: int) -> np.ndarray:
    """Boolean array marking observed timesteps inside the history region.

    History is identified by position rather than by timestamp. The extracted layout reserves
    `[0, max_history_len_steps)` for pre-admission records, right-justified, and that region is
    exactly what `HISTORY_LEN_STEPS` crops and what the history flags mask -- so a positional
    definition makes cohort membership agree with what the model can actually read. It also
    does not depend on the sign convention of the time axis, which is a separate thing to get
    right and has been wrong before.

    Args:
        masks: (n_episodes, max_ts_len) nonzero at non-padding timesteps.
        max_history_len_steps: Width of the history region in the extracted arrays.

    Returns:
        (n_episodes, max_history_len_steps) boolean array.
    """
    if max_history_len_steps <= 0:
        return np.zeros((np.asarray(masks).shape[0], 0), dtype=bool)
    return np.asarray(masks)[:, :max_history_len_steps] > 0


def has_historical_text(val_masks, val_text_indicators, max_history_len_steps: int,
                        feature_index: int) -> np.ndarray:
    """Episodes carrying at least one pre-admission record of one text feature.

    Text records share the value stream's timestep axis, so they are found by intersecting that
    stream's observed history with the feature's presence indicator.

    Returns:
        (n_episodes,) boolean array.
    """
    indicators = np.asarray(val_text_indicators)
    if indicators.shape[2] <= feature_index:
        raise ValueError(
            f'the extraction carries {indicators.shape[2]} text features, so index '
            f'{feature_index} does not exist. Check TEXT_FEATS in the dataset config.'
        )
    observed = history_observed(val_masks, max_history_len_steps)
    present = indicators[:, :max_history_len_steps, feature_index] > 0
    return (observed & present).any(axis=1)


def has_value_history(val_masks, max_history_len_steps: int) -> np.ndarray:
    """Episodes carrying at least one pre-admission record the model can read.

    Only the value stream is checked, and that is deliberate. `collate_tensorized` slices the
    history region off the event stream before the batch is built, because the THP gates its
    base-intensity term on tensor index 0 and leading history padding silently drops it. So no
    pre-admission event ever reaches the model, and an episode whose only history is an event
    is, to every arm, indistinguishable from an episode with no history at all -- which is the
    dilution a cohort exists to remove.

    Returns:
        (n_episodes,) boolean array.
    """
    return history_observed(val_masks, max_history_len_steps).any(axis=1)


def has_any_history(val_masks, event_masks, max_history_len_steps: int) -> np.ndarray:
    """Episodes carrying at least one pre-admission record in either stream.

    Descriptive rather than selective: the streams are filtered independently at extraction, so
    this counts what the extraction holds. It is not the cohort predicate -- see
    `has_value_history` for why the event stream cannot qualify an episode.

    Returns:
        (n_episodes,) boolean array.
    """
    return (history_observed(val_masks, max_history_len_steps).any(axis=1)
            | history_observed(event_masks, max_history_len_steps).any(axis=1))


def cohort_mask(arrays, cohort: Optional[str]) -> Optional[np.ndarray]:
    """Boolean array selecting the episodes a named cohort keeps.

    Args:
        arrays: Any object exposing `val_masks`, `val_text_indicators` and
            `max_history_len_steps` -- a loaded `MixedDataset`, or the arrays behind one.
        cohort: 'discharge_summary', 'any_history', or None for every episode.

    Returns:
        (n_episodes,) boolean array, or None when `cohort` is None.

    Raises:
        ValueError: If `cohort` is not a known name.
    """
    if cohort is None:
        return None
    if cohort not in COHORTS:
        raise ValueError(f'unknown cohort {cohort!r}; expected one of {COHORTS} or None.')

    def field(name):
        return arrays[name] if isinstance(arrays, dict) else getattr(arrays, name)

    hist = int(field('max_history_len_steps'))
    if cohort == 'discharge_summary':
        return has_historical_text(field('val_masks'), field('val_text_indicators'), hist,
                                   DISCHARGE_SUMMARY_INDEX)
    return has_value_history(field('val_masks'), hist)


def cohort_indices(arrays, cohort: Optional[str]) -> Optional[np.ndarray]:
    """Row indices a named cohort keeps, or None for every episode."""
    mask = cohort_mask(arrays, cohort)
    return None if mask is None else np.flatnonzero(mask).astype(np.int64)
