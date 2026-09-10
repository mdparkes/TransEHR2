"""Cohort predicates over the extracted arrays.

A cohort restricts the experiments to episodes that carry a particular kind of pre-admission
record. The comparison between a model that reads history and one that does not is otherwise
biased by the episodes that have no history to read: those contribute identically to both arms
and dilute the contrast, and the paired tests need the two arms to run on the same episodes.

"Pre-admission" is the era the extraction placed in the history region, which begins
`PREADMISSION_CUTOFF_HOURS` before ICU admission. Every predicate here is positional, so each
one means "before the cutoff" without having to be told where the cutoff is -- the extraction
already decided that when it justified the regions.

The predicates read the extracted arrays rather than the source CSVs, so they select within an
extraction that already exists and cost no re-extraction. Cohort membership is computed on the
full extracted history and does not depend on the runtime `HISTORY_LEN_STEPS` crop -- an
episode does not leave the cohort because a sweep arm stopped showing the model its records.

Selection is per episode, matching `filter_listfiles_by_discharge_summary.py`. An episode with
no pre-admission record is what biases the comparison, whether or not the same patient has
another episode that does.

A cohort can also be given as an explicit list of patient-episode IDs -- see `manifest_mask`.
That is the form to use when membership depends on something the arrays do not carry, such as
whether a feature computed from the source CSVs came out available: a predicate over the
arrays can only approximate it, and two arms of a comparison reading one manifest is a
guarantee where two predicates are an argument. The Charlson analysis works this way, because
an episode belongs to it exactly when it has an index, an age and a sex.

`diagnosis_history` is the array-side proxy for the same thing -- an episode carrying at least
one pre-admission diagnosis-descriptions record, i.e. one earlier admission's discharge
diagnoses. It is not the Charlson cohort and should not be used as one: the extraction may
hold no diagnosis record for an episode whose codes are perfectly available in
`diagnoses.csv`, and a peri-stay-only model reads neither.

`any_text` is the text cohort the experiments run on: an episode carrying a pre-admission
record of *any* text feature. It replaces separate discharge-summary and diagnosis cohorts,
which split a population too small to divide and asked the same question twice. Because text
before the cutoff is itself a pre-admission record, `any_text` is a subset of `any_history` by
construction.

`all_text` is the intersection -- every text feature present -- and is a subset of `any_text`
in turn. It is the population for the carry-forward analysis, which needs the diagnosis list
to decide whether a label was already named and the summary as the input under test, so an
episode missing either cannot contribute a comparison. The two single-feature predicates
remain, for describing the population rather than selecting it.
"""

import os
from typing import Optional

import numpy as np


# Positions in the dataset config's TEXT_FEATS list.
DISCHARGE_SUMMARY_INDEX = 0
DIAGNOSIS_DESCRIPTIONS_INDEX = 1

# Valid values for an experiment's COHORT_SUBSET, and the populations the analysis scripts
# offer. `any_text` and `any_history` are the two the study runs on.
COHORTS = ('discharge_summary', 'diagnosis_history', 'all_text', 'any_text',
           'any_history')


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


def has_any_historical_text(val_masks, val_text_indicators,
                            max_history_len_steps: int) -> np.ndarray:
    """Episodes carrying at least one pre-admission record of any text feature.

    Reduced over whatever text features the extraction holds rather than over named indices,
    so the predicate follows TEXT_FEATS instead of having to be edited alongside it.

    Args:
        val_masks: (n_episodes, max_ts_len) nonzero at non-padding timesteps.
        val_text_indicators: (n_episodes, max_ts_len, n_text_feats) presence indicators.
        max_history_len_steps: Width of the history region in the extracted arrays.

    Returns:
        (n_episodes,) boolean array.
    """
    indicators = np.asarray(val_text_indicators)
    observed = history_observed(val_masks, max_history_len_steps)
    if indicators.shape[2] == 0:
        return np.zeros(observed.shape[0], dtype=bool)
    present = (indicators[:, :max_history_len_steps, :] > 0).any(axis=2)
    return (observed & present).any(axis=1)


def has_all_historical_text(val_masks, val_text_indicators,
                            max_history_len_steps: int) -> np.ndarray:
    """Episodes carrying a pre-admission record of *every* text feature.

    The intersection rather than the union, and a strict subset of `has_any_historical_text`.
    It is the population for an analysis that needs both kinds of text present -- the
    carry-forward contrast reads the diagnosis list to decide whether a label was already
    named, and the summary is what the arm under test reads -- so an episode missing either
    one cannot contribute a comparison.

    Reduced over whatever text features the extraction holds, so it follows TEXT_FEATS. An
    extraction with no text features returns all-False rather than the vacuous all-True: no
    episode carries text there, which is what a caller asking this question means.

    Args:
        val_masks: (n_episodes, max_ts_len) nonzero at non-padding timesteps.
        val_text_indicators: (n_episodes, max_ts_len, n_text_feats) presence indicators.
        max_history_len_steps: Width of the history region in the extracted arrays.

    Returns:
        (n_episodes,) boolean array.
    """
    indicators = np.asarray(val_text_indicators)
    observed = history_observed(val_masks, max_history_len_steps)
    n_features = indicators.shape[2]
    if n_features == 0:
        return np.zeros(observed.shape[0], dtype=bool)
    present = np.ones(observed.shape[0], dtype=bool)
    for feature in range(n_features):
        recorded = indicators[:, :max_history_len_steps, feature] > 0
        present &= (observed & recorded).any(axis=1)
    return present


def has_value_history(val_masks, max_history_len_steps: int) -> np.ndarray:
    """Episodes carrying at least one pre-admission record the model can read.

    Only the value stream is checked. `collate_tensorized` slices the history region off the
    event stream before the batch is built -- the THP gates its base-intensity term on tensor
    index 0, and leading history padding silently drops it -- so no pre-admission event reaches
    the model and an episode whose history the model cannot read is the dilution a cohort
    exists to remove.

    Since every event feature is also a valued feature, an event record's timestep appears in
    the value stream too, and this agrees with `has_any_history` except at one boundary: the
    event stream is not resampled, so an event inside the hour that rounds past the cutoff sits
    in the history region of its own stream and in the episode region of the value stream.

    Returns:
        (n_episodes,) boolean array.
    """
    return history_observed(val_masks, max_history_len_steps).any(axis=1)


def has_any_history(val_masks, event_masks, max_history_len_steps: int) -> np.ndarray:
    """Episodes carrying at least one pre-admission record in either stream.

    Descriptive rather than selective: the streams are filtered independently at extraction, so
    this counts what the extraction holds rather than what a batch carries. `has_value_history`
    is the cohort predicate, and it is what the model can actually read; the two now differ only
    at the boundary hour described there.

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
        cohort: A name in `COHORTS`, or None for every episode.

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
    if cohort == 'any_text':
        return has_any_historical_text(field('val_masks'), field('val_text_indicators'), hist)
    if cohort == 'all_text':
        return has_all_historical_text(field('val_masks'), field('val_text_indicators'), hist)
    text_index = {'discharge_summary': DISCHARGE_SUMMARY_INDEX,
                  'diagnosis_history': DIAGNOSIS_DESCRIPTIONS_INDEX}.get(cohort)
    if text_index is not None:
        return has_historical_text(field('val_masks'), field('val_text_indicators'), hist,
                                   text_index)
    return has_value_history(field('val_masks'), hist)


def load_episode_manifest(manifest) -> np.ndarray:
    """The patient-episode IDs an explicit cohort manifest names.

    Args:
        manifest: Path to a file of one integer ID per line -- blank lines and `#` comments
            ignored -- or an iterable of IDs.

    Returns:
        Sorted array of unique IDs.

    Raises:
        FileNotFoundError: If a path is given and does not exist.
        ValueError: If the manifest names no episodes, or holds something that is not an
            integer ID.
    """
    if isinstance(manifest, (str, bytes, os.PathLike)):
        with open(manifest) as handle:
            entries = [line.split('#', 1)[0].strip() for line in handle]
        entries = [entry for entry in entries if entry]
        try:
            ids = [int(entry) for entry in entries]
        except ValueError as exc:
            raise ValueError(
                f'{manifest} is not a list of patient-episode IDs: {exc}. Expected one '
                f'integer per line, as compute_charlson_index.py --write_cohort writes.'
            ) from exc
        source = str(manifest)
    else:
        ids = [int(entry) for entry in manifest]
        source = 'the given manifest'

    if not ids:
        raise ValueError(
            f'{source} names no episodes. A cohort that matches nothing would train on an '
            f'empty dataset.'
        )
    return np.unique(np.asarray(ids, dtype=np.int64))


def manifest_mask(episode_ids, manifest) -> np.ndarray:
    """Boolean array selecting the rows an explicit manifest names.

    Selecting by ID rather than by a predicate over the arrays is what lets a cohort depend on
    something the arrays do not carry -- the availability of a feature computed from the source
    CSVs, say. Both arms of a comparison reading the same manifest is then the guarantee that
    they run on the same episodes, in place of two predicates that have to be argued to agree.

    Args:
        episode_ids: (n_episodes,) patient-episode IDs, one per row of the extracted arrays.
        manifest: Anything `load_episode_manifest` accepts.

    Returns:
        (n_episodes,) boolean array.

    Raises:
        ValueError: If the manifest selects none of these rows, which means it was built
            against a different extraction.
    """
    ids = np.asarray(episode_ids, dtype=np.int64)
    mask = np.isin(ids, load_episode_manifest(manifest))
    if not mask.any():
        raise ValueError(
            f'the cohort manifest names none of this partition\'s {ids.size} episodes, so it '
            f'was built against a different extraction or a different fold layout.'
        )
    return mask


def cohort_indices(arrays, cohort: Optional[str], episode_ids=None,
                   manifest=None) -> Optional[np.ndarray]:
    """Row indices a cohort keeps, or None for every episode.

    A named cohort and a manifest may be given together, in which case an episode has to
    satisfy both.

    Args:
        arrays: As for `cohort_mask`.
        cohort: A name in `COHORTS`, or None.
        episode_ids: (n_episodes,) patient-episode IDs, required with `manifest`.
        manifest: An explicit episode manifest, or None.

    Returns:
        Ascending row indices, or None when neither restriction is given.

    Raises:
        ValueError: If `manifest` is given without `episode_ids`.
    """
    if manifest is not None and episode_ids is None:
        raise ValueError(
            'a cohort manifest selects rows by patient-episode ID, so episode_ids must be '
            'given alongside it.'
        )

    mask = cohort_mask(arrays, cohort)
    if manifest is not None:
        by_id = manifest_mask(episode_ids, manifest)
        mask = by_id if mask is None else (mask & by_id)
    return None if mask is None else np.flatnonzero(mask).astype(np.int64)
