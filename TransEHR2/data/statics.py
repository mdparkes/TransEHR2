"""Reading time-invariant features back out of the stored `static_data` array.

The stored layout is not the obvious one, and getting it wrong yields a plausible number
rather than an error. Two things to know:

    A categorical static is *allocated* `size` columns but *written* as a single code in the
    first of them, so a feature's column is the cumulative width of the features before it,
    not its position in `STATIC_FEATS`. Age + Gender is four columns wide, and Gender starts
    at column 1.

    That code is 1-based, offset from the lowest key of the feature's `category_map`, and the
    column is left at zero when the value was missing. So zero means missing and is not a
    category.

Anything reading these arrays should come through here rather than reimplement the arithmetic,
for the same reason `compute_static_feat_dims` exists on the writing side: the two must not
drift apart.
"""

from typing import Dict, List

import numpy as np

from TransEHR2.data.preprocessing import compute_static_feat_dims

# Label given to an episode whose stored code is zero, or is not in the category map.
MISSING_LABEL = 'Missing'


def static_offsets(variable_properties: dict, static_feats: List[str],
                   max_token_length: int) -> Dict[str, int]:
    """Column offset of each static feature in the stored `static_data` array.

    Args:
        variable_properties: Parsed `variable_properties.yaml`.
        static_feats: The STATIC_FEATS list from the dataset config, in order.
        max_token_length: Width given to a static text feature.

    Returns:
        Dict mapping feature name to its first column.
    """
    widths = compute_static_feat_dims(variable_properties, static_feats, max_token_length)
    offsets, position = {}, 0
    for feature, width in zip(static_feats, widths):
        offsets[feature] = position
        position += width
    return offsets


def decode_categorical(codes, category_map: dict) -> np.ndarray:
    """Turn the stored codes of a categorical static into their labels.

    Args:
        codes: (n_episodes,) array of stored codes.
        category_map: The feature's `category_map` from `variable_properties.yaml`.

    Returns:
        (n_episodes,) array of label strings; a missing or unrecognised code becomes
        `MISSING_LABEL`.
    """
    first_key = min(int(key) for key in category_map) if category_map else 0
    labels = {int(code) - first_key + 1: str(label) for code, label in category_map.items()}
    return np.array([labels.get(int(code), MISSING_LABEL) for code in np.asarray(codes)],
                    dtype=object)


def age_observed(ages) -> np.ndarray:
    """Boolean array marking episodes whose age was recorded.

    A missing numeric static is stored as zero, and the extraction admits adults only, so a
    non-positive age is missing rather than a newborn.

    Args:
        ages: (n_episodes,) stored ages.

    Returns:
        (n_episodes,) boolean array.
    """
    return np.asarray(ages, dtype=np.float64) > 0.0
