"""Stage C extraction: ``data/root`` to ``data/extracted`` (sections 4, 5).

Readings this module commits to
-------------------------------

* **``lookup`` replaces the text internals.** ``_bucket_valued_feats``
  returns numeric / categorical / ordinal / lookup, and text and drugs
  share one CSR writer parameterised by slot count (section 4.3, escalated
  in section 9 and decided 2026 August 23, C1). On disk the names stay
  ``text_*`` and ``drug_*`` per section 4.4; only the internals are one
  path. A single-slot feature writes no dose or mask array, its weight
  being 1 by definition, and its values array is rank 1 -- both are what
  section 4.4's table asks for.
* **The extractor assigns the ``text_values`` row indices** (escalated
  2026 August 23, C1, since section 9 makes neither C1 nor C4 depend on
  the other). Workers hand back the raw string; the insertion pass interns
  it, first appearance in canonical row order winning index 0, and writes
  the table's key order to ``text_strings.pkl``. C4's ``embed_text.py``
  embeds that list *in that order*, which is what makes the index valid.
* **The drug pad index comes from ``ClinVec_atc.csv``, not from
  ``drug_embeddings.npy``.** Section 4.4 pads unused slots with ``V``, the
  final all-zero row of a ``(V+1, 128)`` table C4 builds. Deriving ``V``
  from the cohort's own indices would be wrong -- a vocabulary entry no
  patient was dispensed would leave the pad colliding with a real drug --
  so it is read from the vocabulary file Stage A indexed against
  (``prepare_RMT23345_PIN.R:75-77``). This adds ``CLINVEC_PATH`` to the
  dataset config, which section 5's cleanup list does not mention.
* **Every row of ``labels.csv`` is extracted; none is filtered out.**
  Section 3's fold arrays are row indices into these arrays, so the row
  count is fixed by ``labels.csv`` and a dropped episode would silently
  shift every later fold index. ``MIN_EPISODE_LEN_STEPS`` therefore
  becomes a *reported count* rather than a filter -- invariant 4 makes
  every ``INDEX_TIME`` name a ``timeseries.csv`` row, so an episode below
  the minimum means an upstream invariant failed, which is worth saying
  out loud rather than quietly deleting.
* **The series is right-aligned** (section 4.2): the index time, the last
  record and ``t = 0`` all land in the final column, and the left of the
  row is zero padding. The section's "no left-padding" refers to the
  history region the old layout reserved, which is gone; "right-aligned at
  the prediction origin and zero-padded on the left" is the alignment.
* **Misses are counted once per record, not once per episode.** A record
  inside ten episodes' windows is one out-of-domain value in the data, not
  ten, so the per-feature report (section 4.3) counts over each patient's
  timeseries once.
* **``index_times`` is looked up in ``stays.csv``, in ``labels.csv``
  order.** Section 3 makes ``labels.csv`` the canonical order but gives it
  no ``INDEX_TIME`` column, so the value is joined on
  ``(PATID, STAY_INDEX)`` while the order stays the file's.
* **Extraction loads no tokenizer.** Under section 4.4's content
  addressing the tokenizer only matters when ``text_tokens.npy`` is built,
  which is C4's. ``metadata.pkl`` records ``LLM_NAME``,
  ``TOKENIZER_PAD_TOKEN`` and ``MAX_TOKEN_LENGTH`` from ``constants.py``;
  the *resolved* ``pad_token_id`` section 4.4 also asks for is added by C4,
  where the tokenizer is actually loaded and the fact established.
* **Invariant 12 runs before any patient data is read.** Section 6 asks
  for it as the first step of ``extract_data.py``. The category_map clause
  used to be checked inside the per-timestep loop, so a bad map raised
  part-way through a multi-hour run (section A.3); it is hoisted here.
  IBDdataprep's ``check_contract.py`` is the CI-side copy of the same
  three clauses -- importing it would put a cross-repository dependency in
  the extractor's import path.
"""

import gc
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
import yaml

from collections import Counter
from functools import partial
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, Iterator, List, Optional, Tuple, Union

from TransEHR2.constants import HF_API_TOKEN, LLM_NAME, MAX_TOKEN_LENGTH, TOKENIZER_PAD_TOKEN
from TransEHR2.data.custom_types import EpisodeData, MixedTensorDataset, TensorDimensions
from TransEHR2.data.datareaders import EHRDataReader
from TransEHR2.data.datasets import MixedDataset


# Global variables for multi-process data extraction
_tensorized_processor = None
_tensorized_dims = None

# The two ``type`` values that make a feature a member of the lookup family
# (section 4.3). The on-disk file prefix is the type itself, which is what
# keeps ``text_*`` and ``drug_*`` distinct for the webapp.
LOOKUP_TYPES = ('text', 'drug')

# Index meaning "observed, but the value is not in category_map"
# (section 4.3). Not a magic zero: zero is a real category.
OUT_OF_DOMAIN = -1


def _bucket_valued_feats(
    valued_feats: List[str],
    lookup_feats: List[str],
    var_properties: dict
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split features into numeric, categorical, ordinal, and lookup.

    Each feature is assigned to a bucket by its 'type' in
    variable_properties.yaml. Relative order within each bucket follows
    the order of the input list. Both the tensor-dimension computation and
    the DataProcessor rely on this identical logic, so the array slot for
    a value maps back to its feature via (input order, type).

    ``lookup_feats`` comes from ``TEXT_FEATS + DRUG_FEATS`` rather than
    from ``VALUED_FEATS``: section 2.6 gives ``DRUG_FEATS`` no timeseries
    column at all, so the family's members are named by their own config
    lists, not by a type within the valued list.

    Args:
        valued_feats: Value-associated feature names.
        lookup_feats: Text and drug feature names, text first.
        var_properties: Parsed variable_properties.yaml.

    Returns:
        Tuple of (numeric, categorical, ordinal, lookup) feature names.
    """
    numeric_feats = []
    categorical_feats = []
    ordinal_feats = []
    lookup = []
    for feat in valued_feats:
        feat_type = var_properties[feat]['type']
        if feat_type == 'numeric':
            numeric_feats.append(feat)
        elif feat_type == 'categorical':
            categorical_feats.append(feat)
        elif feat_type == 'ordinal':
            ordinal_feats.append(feat)
    for feat in lookup_feats:
        if var_properties[feat]['type'] in LOOKUP_TYPES:
            lookup.append(feat)
    return numeric_feats, categorical_feats, ordinal_feats, lookup


def check_feature_contract(config: dict, var_properties: dict) -> List[str]:
    """Invariant 12, checked before any patient data is read (section 6).

    Three clauses: the YAML's key set is exactly the union of the config's
    five feature lists; each family's indicator width is the count of
    ``VALUED_FEATS`` of that type; and every categorical or ordinal entry
    has ``size == len(category_map)`` with keys running ``0 .. size-1``.

    Returns:
        A list of failure messages, empty when the contract holds.
    """
    feature_lists = ('VALUED_FEATS', 'EVENT_FEATS', 'TEXT_FEATS',
                     'DRUG_FEATS', 'STATIC_FEATS')
    declared = set()
    for key in feature_lists:
        declared |= set(config.get(key) or [])
    described = set(var_properties)

    failures = []
    missing = sorted(declared - described)
    if missing:
        failures.append(
            f"{len(missing)} config feature(s) have no "
            f"variable_properties entry: {missing}"
        )
    stale = sorted(described - declared)
    if stale:
        failures.append(
            f"{len(stale)} variable_properties entr(ies) appear in no "
            f"config feature list: {stale}"
        )

    for feat in sorted(declared & described):
        entry = var_properties[feat]
        if entry.get('type') not in ('categorical', 'ordinal'):
            continue
        cat_map = entry.get('category_map') or {}
        size = entry.get('size')
        keys = sorted(int(k) for k in cat_map)
        if len(cat_map) != size or keys != list(range(size)):
            failures.append(
                f"feature '{feat}': size {size} but category_map keys "
                f"{keys[:5]}{'...' if len(keys) > 5 else ''} "
                f"({len(cat_map)} of them); keys must run 0..size-1"
            )
    return failures


class LLMTextProcessor:

    def __init__(
        self,
        model_name: str = LLM_NAME,
        max_length: int = MAX_TOKEN_LENGTH
    ):
        """
        Initialize the LLM text processor.

        Args:
            model_name (str): The LLM model name to use for tokenization
            max_length (int): Maximum sequence length for tokenized text
        """

        # Use the Llama-3.1-8B tokenizer explicitly because the
        # Llama-3.2-1B tokenizer has a broken tekken.json path that
        # causes AttributeError in convert_slow_tokenizer. The
        # tokenizer vocabulary is compatible across Llama 3.x models.
        name_or_basename = model_name + '/' + os.path.basename(model_name.rstrip('/'))
        if 'Llama-3.2-1B' in name_or_basename:
            if os.environ.get('HF_HUB_OFFLINE', '0') == '1':
                tokenizer_name = os.path.join(os.path.dirname(model_name), 'Llama-3.1-8B')
            else:
                tokenizer_name = 'meta-llama/Llama-3.1-8B'
        else:
            tokenizer_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=HF_API_TOKEN,
                local_files_only=True,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=HF_API_TOKEN,
            )
        self.tokenizer.add_special_tokens(
            {'pad_token': TOKENIZER_PAD_TOKEN}
        )
        self.max_length = max_length


    def process_text(self, text: str) -> Dict[str, np.ndarray]:
        """
        Process a single text string and convert it to token IDs.

        Args:
            text (str): A single text string to tokenize

        Returns:
            numpy.ndarray: Array of token IDs with shape (max_tokens,)
        """
        # An empty string is all padding, not all token 0. Section 4.4
        # derives the attention mask as ``ids != pad_id`` and stores no
        # mask, so the returned pair has to be self-consistent even on
        # this branch -- with zeros it would invert to an all-ones mask.
        if not text or pd.isna(text) or text.strip() == '':
            pad_id = self.tokenizer.pad_token_id
            return {
                'input_ids': np.full(
                    self.max_length, pad_id, dtype=np.int32
                ),
                'attention_mask': np.zeros(self.max_length, dtype=np.int32)
            }

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'  # Return numpy arrays
        )

        # Return a dictionary with 'input_ids' and 'attention_mask'
        return {
            'input_ids': tokenized['input_ids'][0],  # remove batch dimension
            'attention_mask': tokenized['attention_mask'][0]  # remove batch dimension
        }


class DataProcessor:
    """
    Processes one patient's record into numpy arrays for tensor insertion.

    The unit is the patient, not the episode (section 4.1): a patient's
    episodes are nested suffixes of one timeseries, so the columns are
    typed and mapped once and each episode is a slice of the result.

    The processor handles:
    - Numeric features: float32 values, zero where unobserved
    - Categorical and ordinal features: int16 category indices, -1 for
      unobserved or out-of-domain (section 4.3)
    - Lookup features: CSR entries carrying a string (text) or slot
      indices with doses and masks (drugs)
    - Event indicators: float32, set from presence alone
    - Static features: concatenated float32 array
    """

    def __init__(
        self,
        var_properties_path: str,
        valued_feats: List[str],
        event_feats: List[str],
        lookup_feats: List[str],
        static_feats: List[str],
        dims: TensorDimensions
    ):
        """
        Initialize the data processor.

        Args:
            var_properties_path: Path to variable_properties.yaml
            valued_feats: List of value-associated feature names
            event_feats: List of event-associated feature names
            lookup_feats: Text and drug feature names, text first
            static_feats: List of static feature names
            dims: Pre-computed tensor dimensions
        """
        with open(var_properties_path, 'r') as f:
            self.var_properties = yaml.safe_load(f)

        (self.numeric_feats,
         self.categorical_feats,
         self.ordinal_feats,
         self.lookup_feats) = _bucket_valued_feats(
            valued_feats, lookup_feats, self.var_properties
        )

        self.event_feats = event_feats
        self.static_feats = static_feats
        self.dims = dims

        # value -> index, keyed on ``str`` so that the dtype trap in
        # section 5 cannot fire: BLDUA's levels are numeric-looking
        # strings, and a patient whose column pandas read as numeric would
        # otherwise miss every one of them.
        self.index_maps = {
            feat: {
                str(v): int(k)
                for k, v in (self.var_properties[feat].get('category_map')
                             or {}).items()
            }
            for feat in self.categorical_feats + self.ordinal_feats
        }

    def process_valued_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray],
               np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Process one patient's value-associated data.

        Args:
            data: DataFrame indexed by absolute timestamp, one row per
                distinct minute (section 2.6).

        Returns:
            Tuple of:
            - numeric_indicators: (n_ts, n_numeric_feats) float32
            - numeric_values: List of (n_ts, feat_dim) float32
            - categorical_indicators: (n_ts, n_cat_feats) float32
            - categorical_values: List of (n_ts,) int16 indices
            - categorical_misses: (n_cat_feats,) int64
            - ordinal_indicators: (n_ts, n_ord_feats) float32
            - ordinal_values: List of (n_ts,) int16 indices
            - ordinal_misses: (n_ord_feats,) int64
        """
        n_ts = len(data)

        numeric_indicators = np.zeros(
            (n_ts, self.dims.n_numeric_feats), dtype=np.float32
        )
        numeric_values = [
            np.zeros((n_ts, dim), dtype=np.float32)
            for dim in self.dims.numeric_feat_dims
        ]
        for f, feat in enumerate(self.numeric_feats):
            cols = self._feature_columns(feat, data)
            if not cols:
                continue
            block = data[cols].to_numpy(dtype=np.float32)
            observed = ~np.isnan(block).all(axis=1)
            numeric_indicators[:, f] = observed
            # Every numeric feature is size 1 today (section 4.3), but a
            # vector one must not write past its declared dimension.
            width = min(self.dims.numeric_feat_dims[f], block.shape[1])
            numeric_values[f][:, :width] = np.nan_to_num(
                block[:, :width], nan=0.0
            )

        (categorical_indicators, categorical_values,
         categorical_misses) = self._process_indexed(
            data, self.categorical_feats, self.dims.n_categorical_feats
        )
        (ordinal_indicators, ordinal_values,
         ordinal_misses) = self._process_indexed(
            data, self.ordinal_feats, self.dims.n_ordinal_feats
        )

        return (numeric_indicators, numeric_values,
                categorical_indicators, categorical_values,
                categorical_misses,
                ordinal_indicators, ordinal_values, ordinal_misses)

    def _process_indexed(
        self, data: pd.DataFrame, feats: List[str], n_feats: int
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Map categorical or ordinal values to int16 indices.

        Indicator and index are independent (section 4.3): indicator 0
        with index -1 is "not observed", indicator 1 with index -1 is
        "observed, out of domain", indicator 1 with index k is category k.
        The arrays start at -1, never at 0 -- zero is ``"L"`` for
        ``ADMITCAT`` and level ``"0"`` for ``BLDUA``, so a zero-filled
        array hands the loss a false label rather than an absent one.
        """
        n_ts = len(data)
        indicators = np.zeros((n_ts, n_feats), dtype=np.float32)
        values = [
            np.full(n_ts, OUT_OF_DOMAIN, dtype=np.int16)
            for _ in range(n_feats)
        ]
        misses = np.zeros(n_feats, dtype=np.int64)

        for f, feat in enumerate(feats):
            if feat not in data.columns:
                continue
            column = data[feat]
            observed = column.notna().to_numpy()
            indicators[:, f] = observed
            # ``astype(str)`` renders a missing cell as a *string* -- 'nan'
            # for a NaN, and 'None' for the None that ``groupby.first()``
            # leaves in an object column. 'None' is UBAC's declared level
            # 0, so an unobserved cell would map straight onto it: a
            # false label with indicator 0, the exact confusion section
            # 4.3's sentinel exists to prevent. Masking the mapped result
            # back to NaN wherever the cell was not observed makes the
            # lookup independent of how the missing value renders.
            mapped = column.astype(str).map(self.index_maps[feat])
            mapped = mapped.where(observed)
            values[f] = mapped.fillna(OUT_OF_DOMAIN).to_numpy(np.int16)
            misses[f] = int((observed & mapped.isna().to_numpy()).sum())

        return indicators, values, misses

    def process_lookup_data(
        self, text_data: pd.DataFrame, drug_data: pd.DataFrame,
        timestamps: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, List[list]]:
        """
        Build the lookup family's indicators and CSR entries.

        Text and drugs are one family (section 4.3) but reach the reader
        by different routes: a text feature is a column of
        ``timeseries.csv``, while ``DRUG_FEATS`` has no column at all
        (section 2.6) and arrives as ``drugs.csv`` rows joined on
        ``TIMESTAMP``.

        Args:
            text_data: Text columns, indexed by timestamp.
            drug_data: ``drugs.csv`` rows with SLOT >= 0.
            timestamps: The patient's timestep index.

        Returns:
            Tuple of:
            - indicators: (n_ts, n_lookup_feats) float32
            - sparse: per-feature list of
              ``(timestep, values, doses, masks)``, values being the raw
              string for a text feature and an int32 (n_slots,) array for
              a drug feature.
        """
        n_ts = len(timestamps)
        indicators = np.zeros(
            (n_ts, self.dims.n_lookup_feats), dtype=np.float32
        )
        sparse = [[] for _ in range(self.dims.n_lookup_feats)]

        for f, feat in enumerate(self.lookup_feats):
            feat_type = self.var_properties[feat]['type']
            if feat_type == 'text':
                entries = self._text_entries(text_data, feat)
            else:
                entries = self._drug_entries(
                    drug_data, timestamps, self.dims.lookup_slot_dims[f],
                    self.dims.lookup_pad_indices[f]
                )
            for entry in entries:
                indicators[entry[0], f] = 1.0
            sparse[f] = entries

        return indicators, sparse

    @staticmethod
    def _text_entries(text_data: pd.DataFrame, feat: str) -> list:
        """One entry per timestep whose text cell is non-blank.

        A blank cell is not a text record: section 2.6 gives it indicator
        0 and no CSR entry, and section 4.4 keeps the empty string out of
        the unique-string table for the same reason.
        """
        if feat not in text_data.columns:
            return []
        column = text_data[feat]
        present = column.notna().to_numpy()
        stripped = column.fillna('').astype(str).str.strip()
        present &= (stripped != '').to_numpy()
        return [
            (int(t), str(column.iloc[t]), None, None)
            for t in np.flatnonzero(present)
        ]

    @staticmethod
    def _drug_entries(
        drug_data: pd.DataFrame, timestamps: pd.DatetimeIndex,
        n_slots: int, pad_index: int
    ) -> list:
        """One entry per timestep carrying a dispensation.

        Unused slots take ``pad_index`` -- the all-zero final row of
        ``drug_embeddings.npy`` (section 4.4) -- with dose 0 and mask 0.
        ``drug_masks`` is derivable from ``values == pad_index`` and is
        kept anyway, per section 4.4.

        A ``drugs.csv`` timestamp naming no ``timeseries.csv`` row would
        violate invariant 4, so it is an error rather than a dropped
        dispensation.
        """
        if drug_data.empty:
            return []
        positions = timestamps.get_indexer(drug_data['TIMESTAMP'])
        if (positions < 0).any():
            unmatched = drug_data['TIMESTAMP'][positions < 0].unique()
            raise ValueError(
                f"invariant 4: {len(unmatched)} drugs.csv timestamp(s) "
                f"name no timeseries.csv row, e.g. {unmatched[0]}"
            )

        entries = []
        for t, rows in pd.DataFrame({
            'position': positions,
            'slot': drug_data['SLOT'].to_numpy(dtype=int),
            'index': drug_data['CLINVEC_INDEX'].to_numpy(dtype=np.int32),
            'dose': drug_data['REL_DAILY_QTY'].to_numpy(dtype=np.float32),
        }).groupby('position'):
            values = np.full(n_slots, pad_index, dtype=np.int32)
            doses = np.zeros(n_slots, dtype=np.float32)
            masks = np.zeros(n_slots, dtype=np.float32)
            slots = rows['slot'].to_numpy()
            if (slots >= n_slots).any():
                raise ValueError(
                    f"drugs.csv SLOT {slots.max()} exceeds the "
                    f"{n_slots} slots declared for the feature"
                )
            values[slots] = rows['index'].to_numpy()
            doses[slots] = np.nan_to_num(rows['dose'].to_numpy(), nan=0.0)
            masks[slots] = 1.0
            entries.append((int(t), values, doses, masks))
        return entries

    def process_event_data(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Process event-associated data into a flat numpy array.

        The indicator is set from *presence* -- any non-NA, non-empty cell
        -- and the declared type is never consulted (section A.3). This is
        why section 2.6 requires ``ADMIT_DAD`` to be empty rather than 0
        on a non-admission row.

        Args:
            data: DataFrame indexed by timestamp, event columns only

        Returns:
            indicators: (n_timesteps, n_event_feats) float32
        """
        n_ts = len(data)
        indicators = np.zeros(
            (n_ts, self.dims.n_event_feats), dtype=np.float32
        )
        if n_ts == 0:
            return indicators

        for f, feat in enumerate(self.event_feats):
            cols = self._feature_columns(feat, data)
            if not cols:
                continue
            present = data[cols].notna()
            for col in cols:
                if data[col].dtype == object:
                    present[col] &= (
                        data[col].fillna('').astype(str).str.strip() != ''
                    )
            indicators[:, f] = present.any(axis=1).to_numpy()

        return indicators

    def process_static_data(
        self,
        data: Union[pd.Series, pd.DataFrame, None]
    ) -> np.ndarray:
        """
        Process static data into a flat numpy array.

        ``STATIC_FEATS`` is empty under section A.3, so this returns a
        zero-width array and section 4.4 writes no file. The path is kept
        rather than deleted so that restoring a static feature does not
        reach into the encoder.

        Args:
            data: Series or DataFrame containing static features

        Returns:
            Concatenated array of shape (static_total_dim,) as float32
        """
        static_array = np.zeros(self.dims.static_total_dim, dtype=np.float32)

        if data is None or (hasattr(data, 'empty') and data.empty):
            return static_array

        if isinstance(data, pd.DataFrame):
            data = data.iloc[0]

        offset = 0
        for f, feat in enumerate(self.static_feats):
            feat_dim = self.dims.static_feat_dims[f]
            feat_type = self.var_properties[feat]['type']

            if feat in data.index:
                value = data[feat]

                if feat_type == 'numeric':
                    if pd.notna(value):
                        static_array[offset] = float(value)

                elif feat_type in ('categorical', 'ordinal'):
                    if pd.notna(value):
                        static_array[offset] = float(
                            self.index_maps.get(feat, {}).get(
                                str(value), OUT_OF_DOMAIN
                            )
                        )

            offset += feat_dim

        return static_array

    @staticmethod
    def _feature_columns(base_name: str, data: pd.DataFrame) -> List[str]:
        """Find the columns of a scalar or vector-valued feature."""
        if base_name in data.columns:
            return [base_name]
        return [
            col for col in data.columns
            if col.startswith(f'{base_name}_')
            and col[len(base_name) + 1:].isdigit()
        ]


class TextBalancedDistributedSampler(Sampler):
    """
    Distributed sampler that balances text density across ranks.

    Within each meta-batch (batch_size * world_size samples), episodes are
    sorted by text density and assigned to ranks via round-robin, ensuring
    each rank gets a mix of text-heavy and text-light episodes.

    This prevents memory imbalance where one GPU gets all text-heavy episodes
    and OOMs while others sit idle with light batches.

    Randomness is preserved through:
    - Global shuffle at start of each epoch
    - Different episodes grouped into meta-batches each epoch
    - Only the within-meta-batch distribution is deterministic
    """

    def __init__(
        self,
        dataset,
        text_counts: np.ndarray,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        """
        Args:
            dataset: The dataset to sample from
            text_counts: Array of shape (n_episodes,) with text entry count per episode
            batch_size: Per-GPU batch size
            num_replicas: Number of distributed processes (defaults to world size)
            rank: Rank of current process (defaults to current rank)
            shuffle: Whether to shuffle indices each epoch
            seed: Random seed for shuffling
            drop_last: Whether to drop incomplete final meta-batch
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            if torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, must be in [0, {num_replicas})")

        self.dataset = dataset
        self.text_counts = np.asarray(text_counts)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate number of samples per replica
        self.meta_batch_size = batch_size * num_replicas
        self.total_size = len(dataset)

        if self.drop_last and self.total_size % self.meta_batch_size != 0:
            # Number of complete meta-batches
            self.num_meta_batches = self.total_size // self.meta_batch_size
            self.num_samples = (self.num_meta_batches * self.meta_batch_size) // self.num_replicas
        else:
            # Pad to make evenly divisible
            self.num_meta_batches = (self.total_size + self.meta_batch_size - 1) // self.meta_batch_size
            self.num_samples = (self.num_meta_batches * self.meta_batch_size) // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # Create generator with seed + epoch for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Get all indices
        n = len(self.dataset)

        if self.shuffle:
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        # Pad if necessary to make evenly divisible by meta_batch_size
        if not self.drop_last:
            padding_size = self.num_meta_batches * self.meta_batch_size - n
            if padding_size > 0:
                # Pad with repeated indices from the beginning
                indices += indices[:padding_size]
        else:
            # Truncate to complete meta-batches only
            indices = indices[:self.num_meta_batches * self.meta_batch_size]

        # Balance within each meta-batch and extract this rank's indices
        balanced_indices = []

        for start in range(0, len(indices), self.meta_batch_size):
            meta_batch = indices[start:start + self.meta_batch_size]

            if len(meta_batch) < self.meta_batch_size:
                # Incomplete meta-batch at end (shouldn't happen with padding, but safety check)
                if self.drop_last:
                    continue

            # Sort meta-batch by text density (ascending: lightest first)
            meta_batch_sorted = sorted(meta_batch, key=lambda i: self.text_counts[i])

            # Pair lightest with heaviest to balance each rank's total text load
            # E.g., for 16 items [0..15] sorted by density, create pairs:
            #   (0, 15), (1, 14), (2, 13), ..., (7, 8)
            # Then assign pairs round-robin to ranks so each rank gets balanced load
            n_items = len(meta_batch_sorted)
            n_pairs = n_items // 2

            # Build pairs: (lightest, heaviest), (2nd lightest, 2nd heaviest), ...
            pairs = []
            for i in range(n_pairs):
                light_idx = meta_batch_sorted[i]
                heavy_idx = meta_batch_sorted[n_items - 1 - i]
                pairs.append((light_idx, heavy_idx))

            # Handle odd item (middle element) if present
            middle_item = None
            if n_items % 2 == 1:
                middle_item = meta_batch_sorted[n_pairs]

            # Assign pairs to ranks using snake/boustrophedon pattern
            # This reverses direction each pass through ranks to balance any
            # systematic bias from pair ordering
            # E.g., with 4 ranks and 8 pairs:
            #   Pass 0: pair 0->rank 0, pair 1->rank 1, pair 2->rank 2, pair 3->rank 3
            #   Pass 1: pair 4->rank 3, pair 5->rank 2, pair 6->rank 1, pair 7->rank 0
            for pair_idx, (light_idx, heavy_idx) in enumerate(pairs):
                pass_num = pair_idx // self.num_replicas
                pos_in_pass = pair_idx % self.num_replicas

                if pass_num % 2 == 0:
                    # Forward pass: rank 0, 1, 2, ...
                    assigned_rank = pos_in_pass
                else:
                    # Reverse pass: rank n-1, n-2, ..., 0
                    assigned_rank = self.num_replicas - 1 - pos_in_pass

                if assigned_rank == self.rank:
                    balanced_indices.append(light_idx)
                    balanced_indices.append(heavy_idx)

            # Assign middle item (if any) to rank based on number of passes
            # Alternate which rank gets the middle item for additional balancing
            if middle_item is not None:
                n_passes = (n_pairs + self.num_replicas - 1) // self.num_replicas
                middle_rank = n_passes % self.num_replicas
                if self.rank == middle_rank:
                    balanced_indices.append(middle_item)

        return iter(balanced_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch


def get_text_counts_from_dataset(dataset) -> np.ndarray:
    """
    Compute total text entry count per episode from MixedDataset sparse storage.

    Args:
        dataset: MixedDataset instance with sparse text storage

    Returns:
        Array of shape (n_episodes,) with total text entries per episode
    """
    n_episodes = dataset.n_episodes
    text_counts = np.zeros(n_episodes, dtype=np.int32)

    # Sum across all text features
    for f in range(dataset.n_text_feats):
        offsets = dataset.val_text_offsets[f]
        # offsets[i+1] - offsets[i] gives count for episode i
        for i in range(n_episodes):
            text_counts[i] += int(offsets[i + 1]) - int(offsets[i])

    return text_counts


def _init_tensorized_worker(
    var_properties_path: str,
    valued_feats: List[str],
    event_feats: List[str],
    lookup_feats: List[str],
    static_feats: List[str],
    dims_dict: dict
):
    """
    Initialize worker process with a DataProcessor.

    Called once per worker when the process pool is created. No tokenizer
    is built: extraction stores strings and indices, and C4 embeds them.
    """
    global _tensorized_processor, _tensorized_dims

    _tensorized_dims = TensorDimensions(**dims_dict)

    _tensorized_processor = DataProcessor(
        var_properties_path=var_properties_path,
        valued_feats=valued_feats,
        event_feats=event_feats,
        lookup_feats=lookup_feats,
        static_feats=static_feats,
        dims=_tensorized_dims
    )


def filter_timeseries_records(
    index: pd.DatetimeIndex,
    index_time: pd.Timestamp,
    max_episode_len_steps: int
) -> Tuple[int, int]:
    """Select an episode's records: the **last** N at or before the origin.

    This is the opposite end from MIMIC (section 4.1). The MIMIC filter
    kept the *first* N records of a window, which is right for "the first
    48 h of an ICU stay"; an IBD episode is every record up to its
    ``INDEX_TIME``, of which the most recent ``max_episode_len_steps`` are
    kept, because the prediction is made looking backwards from the origin.

    ``index`` must be sorted, which the reader guarantees, so the kept
    records are a contiguous slice and no boolean mask is needed.

    Args:
        index: The patient's timestep index, ascending.
        index_time: The episode's prediction origin.
        max_episode_len_steps: Most records to keep.

    Returns:
        ``(start, stop)`` bounds of the slice to keep. Empty when the
        patient has no record at or before the origin.
    """
    stop = int(index.searchsorted(index_time, side='right'))
    start = max(0, stop - max_episode_len_steps)
    return start, stop


def _process_single_patient(
    i: int,
    reader: EHRDataReader,
    max_episode_len_steps: int
) -> Tuple[List[EpisodeData], np.ndarray, np.ndarray, Optional[str]]:
    """
    Process every episode of one patient (section 4.1).

    One CSV read, one typing pass and one mapping pass serve all of the
    patient's episodes, each of which is a suffix slice of the result.

    Args:
        i: Index of the patient in the reader
        reader: EHRDataReader instance
        max_episode_len_steps: Most timesteps to keep per episode

    Returns:
        ``(episodes, categorical_misses, ordinal_misses, error)``. On
        failure the episode list is empty and ``error`` describes it; the
        caller counts that as a failed patient rather than losing rows
        silently.
    """
    global _tensorized_processor, _tensorized_dims
    dims = _tensorized_dims

    try:
        (patid, episode_rows, statics, val_data, event_data,
         text_data, drug_data) = reader[i]

        # One row per distinct minute. Invariant 3 already guarantees it,
        # so this is identity on conforming input; it replaces
        # ``.resample('1h')``, which materializes every hourly bin between
        # a patient's first and last record -- ~175,200 empty bins across
        # a 20-year history, per episode, per worker (section 5).
        val_data = val_data.groupby(val_data.index.floor('min')).first()
        text_data = text_data.groupby(text_data.index.floor('min')).first()
        if not event_data.empty:
            event_data = event_data.groupby(
                event_data.index.floor('min')
            ).first()

        timestamps = val_data.index

        (num_ind, num_vals, cat_ind, cat_vals, cat_misses,
         ord_ind, ord_vals, ord_misses) = (
            _tensorized_processor.process_valued_data(val_data)
        )
        lookup_ind, lookup_sparse = (
            _tensorized_processor.process_lookup_data(
                text_data, drug_data, timestamps
            )
        )
        event_ind = _tensorized_processor.process_event_data(event_data)
        static_arr = _tensorized_processor.process_static_data(statics)

        episodes = []
        for episode in episode_rows.itertuples(index=False):
            index_time = episode.INDEX_TIME
            start, stop = filter_timeseries_records(
                timestamps, index_time, max_episode_len_steps
            )
            e_start, e_stop = filter_timeseries_records(
                event_data.index, index_time, max_episode_len_steps
            )

            val_times = _minutes_before(
                timestamps[start:stop], index_time
            )
            event_times = _minutes_before(
                event_data.index[e_start:e_stop], index_time
            )

            episodes.append(EpisodeData(
                row=int(episode.ROW),
                val_len=stop - start,
                event_len=e_stop - e_start,
                val_times=val_times,
                val_numeric_indicators=num_ind[start:stop],
                val_numeric_values=[v[start:stop] for v in num_vals],
                val_categorical_indicators=cat_ind[start:stop],
                val_categorical_values=[v[start:stop] for v in cat_vals],
                val_ordinal_indicators=ord_ind[start:stop],
                val_ordinal_values=[v[start:stop] for v in ord_vals],
                val_lookup_indicators=lookup_ind[start:stop],
                val_categorical_misses=cat_misses,
                val_ordinal_misses=ord_misses,
                val_lookup_sparse=[
                    [(t - start, values, doses, masks)
                     for (t, values, doses, masks) in feature_entries
                     if start <= t < stop]
                    for feature_entries in lookup_sparse
                ],
                event_times=event_times,
                event_indicators=event_ind[e_start:e_stop],
                static_data=static_arr,
                time_to_event=float(episode.TIME_TO_EVENT),
                event_type=int(episode.EVENT_TYPE),
                index_time=index_time.tz_localize(None).to_datetime64(),
            ))

        return episodes, cat_misses, ord_misses, None

    except Exception as err:
        n_cat = dims.n_categorical_feats if dims else 0
        n_ord = dims.n_ordinal_feats if dims else 0
        return ([], np.zeros(n_cat, dtype=np.int64),
                np.zeros(n_ord, dtype=np.int64),
                f"patient index {i}: {type(err).__name__}: {err}")


def _minutes_before(
    timestamps: pd.DatetimeIndex, index_time: pd.Timestamp
) -> np.ndarray:
    """Minutes from ``index_time``, negative before it (section 4.2).

    int32 minutes, not float hours or days: float32 has a 24-bit
    significand, exact for integers to 16,777,216 -- +/-31.9 years in
    minutes -- but near t = -10,000 days its spacing is ~0.00098 d against
    a minute's 0.00069 d, so sub-day offsets would collapse entirely.
    """
    if len(timestamps) == 0:
        return np.zeros(0, dtype=np.int32)
    deltas = (timestamps - index_time) // pd.Timedelta(minutes=1)
    return np.asarray(deltas, dtype=np.int32)


def _get_tensor_dimensions(
    var_properties_path: str,
    valued_feats: List[str],
    event_feats: List[str],
    lookup_feats: List[str],
    static_feats: List[str],
    max_ts_len: int,
    n_episodes: int,
    clinvec_path: Optional[str] = None
) -> TensorDimensions:
    """
    Compute tensor dimensions from configuration for pre-allocation.

    The widths are *derived* from the two config files -- section 4.4's
    94 / 37 / 16 are counts of ``VALUED_FEATS`` entries by ``type`` -- and
    are never declared here.

    Args:
        var_properties_path: Path to variable_properties.yaml
        valued_feats: List of value-associated feature names
        event_feats: List of event-associated feature names
        lookup_feats: Text and drug feature names, text first
        static_feats: List of static feature names
        max_ts_len: Timesteps per episode
        n_episodes: Number of rows in labels.csv
        clinvec_path: ``ClinVec_atc.csv``, the vocabulary Stage A indexed
            drugs against. Supplies the drug table's row count -- the pad
            index of section 4.4 -- and its embedding width.

    Returns:
        TensorDimensions dataclass with all dimension information
    """
    with open(var_properties_path, 'r') as f:
        var_properties = yaml.safe_load(f)

    numeric_feats, categorical_feats, ordinal_feats, lookup = (
        _bucket_valued_feats(valued_feats, lookup_feats, var_properties)
    )

    numeric_feat_dims = [var_properties[f]['size'] for f in numeric_feats]
    categorical_feat_dims = [
        var_properties[f]['size'] for f in categorical_feats
    ]
    ordinal_feat_dims = [var_properties[f]['size'] for f in ordinal_feats]

    # ``size`` is the per-timestep dimension for every type (section 4.3);
    # for a lookup feature that is its slot count.
    lookup_slot_dims = [var_properties[f]['size'] for f in lookup]
    lookup_table_dims = []
    lookup_pad_indices = []
    drug_table = _read_clinvec_shape(clinvec_path) if clinvec_path else None
    for feat in lookup:
        if var_properties[feat]['type'] == 'drug':
            if drug_table is None:
                raise ValueError(
                    f"lookup feature '{feat}' has type 'drug' but no "
                    f"CLINVEC_PATH was given; the pad index of section "
                    f"4.4 comes from the vocabulary, not from the cohort"
                )
            n_rows, table_dim = drug_table
            lookup_table_dims.append(table_dim)
            lookup_pad_indices.append(n_rows)
        else:
            # The text table is C4's; a one-slot feature never pads.
            lookup_table_dims.append(None)
            lookup_pad_indices.append(None)

    static_feat_dims = [var_properties[f]['size'] for f in static_feats]

    return TensorDimensions(
        n_episodes=n_episodes,
        max_ts_len=max_ts_len,
        n_numeric_feats=len(numeric_feats),
        n_categorical_feats=len(categorical_feats),
        n_ordinal_feats=len(ordinal_feats),
        n_lookup_feats=len(lookup),
        n_event_feats=len(event_feats),
        numeric_feat_dims=numeric_feat_dims,
        categorical_feat_dims=categorical_feat_dims,
        ordinal_feat_dims=ordinal_feat_dims,
        lookup_slot_dims=lookup_slot_dims,
        lookup_table_dims=lookup_table_dims,
        lookup_pad_indices=lookup_pad_indices,
        static_feat_dims=static_feat_dims,
        static_total_dim=sum(static_feat_dims),
    )


def _read_clinvec_shape(clinvec_path: str) -> Tuple[int, int]:
    """Return ``(n_rows, embedding_dim)`` of the ClinVec ATC vocabulary.

    ``CLINVEC_INDEX`` is the 0-based row index of a drug's ATC code in
    this file (``prepare_RMT23345_PIN.R:75-77``), so ``n_rows`` is the pad
    index ``V`` of section 4.4 and the table C4 builds is
    ``(V+1, embedding_dim)``.
    """
    header = pd.read_csv(clinvec_path, nrows=0).columns
    n_rows = sum(1 for _ in open(clinvec_path, 'rb')) - 1
    return n_rows, len(header) - 1


def _allocate_output_arrays(dims: TensorDimensions) -> Dict[str, np.ndarray]:
    """
    Pre-allocate output arrays as numpy (not torch).

    Categorical and ordinal values start at -1, not 0: section 4.3's
    sentinel means "observed, out of domain", and zero is a real category.
    """
    arrays = {}

    n = dims.n_episodes
    ts = dims.max_ts_len

    arrays['val_times'] = np.zeros((n, ts), dtype=np.int32)
    arrays['val_masks'] = np.zeros((n, ts), dtype=np.float32)

    arrays['val_numeric_indicators'] = np.zeros(
        (n, ts, dims.n_numeric_feats), dtype=np.float32
    )
    arrays['val_numeric_values'] = [
        np.zeros((n, ts, dim), dtype=np.float32)
        for dim in dims.numeric_feat_dims
    ]

    arrays['val_categorical_indicators'] = np.zeros(
        (n, ts, dims.n_categorical_feats), dtype=np.float32
    )
    arrays['val_categorical_values'] = [
        np.full((n, ts), OUT_OF_DOMAIN, dtype=np.int16)
        for _ in range(dims.n_categorical_feats)
    ]

    arrays['val_ordinal_indicators'] = np.zeros(
        (n, ts, dims.n_ordinal_feats), dtype=np.float32
    )
    arrays['val_ordinal_values'] = [
        np.full((n, ts), OUT_OF_DOMAIN, dtype=np.int16)
        for _ in range(dims.n_ordinal_feats)
    ]

    arrays['val_lookup_indicators'] = np.zeros(
        (n, ts, dims.n_lookup_feats), dtype=np.float32
    )

    # Sparse lookup: one bucket per episode row rather than one flat
    # list per feature. CSR wants its entries grouped by row in row
    # order, but results arrive in whatever order the workers finish, so
    # bucketing lets an episode be inserted the moment it arrives and
    # concatenated in canonical order at the end. Holding whole
    # EpisodeData objects until every patient was done would instead keep
    # the dense per-episode arrays alive beside the output arrays -- the
    # dataset twice over, and section 4.5 puts one copy at ~75 GB.
    arrays['_lookup_entries'] = [
        [[] for _ in range(n)] for _ in range(dims.n_lookup_feats)
    ]

    arrays['event_times'] = np.zeros((n, ts), dtype=np.int32)
    arrays['event_masks'] = np.zeros((n, ts), dtype=np.float32)
    arrays['event_indicators'] = np.zeros(
        (n, ts, dims.n_event_feats), dtype=np.float32
    )

    arrays['static_data'] = np.zeros(
        (n, dims.static_total_dim), dtype=np.float32
    )
    arrays['time_to_event'] = np.zeros(n, dtype=np.float32)
    arrays['event_type'] = np.zeros(n, dtype=np.int8)
    arrays['index_times'] = np.zeros(n, dtype='datetime64[ns]')

    return arrays


def _finalize_sparse_lookup(
    arrays: Dict[str, np.ndarray],
    dims: TensorDimensions
) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    """
    Convert the per-row lookup buckets to CSR arrays, one dict each.

    One writer serves both members of the family (section 4.3). A
    single-slot feature gets no ``doses`` or ``masks`` key -- its weight
    is 1 by definition, so both arrays would be constant and section 4.4
    lists neither -- and its values array is rank 1.

    Rows are walked in order, which is where a text string earns its
    index into ``text_embeddings.npy``: first appearance in canonical row
    order wins index 0. That makes the table's key order a function of
    ``labels.csv`` alone, so it does not move with the worker count.

    Returns:
        ``(csr_per_feature, strings)``, ``strings`` being the table's key
        order for C4 to embed.
    """
    finalized = []
    text_table: Dict[str, int] = {}

    for f in range(dims.n_lookup_feats):
        n_slots = dims.lookup_slot_dims[f]
        buckets = arrays['_lookup_entries'][f]
        offsets = np.zeros(len(buckets) + 1, dtype=np.int64)
        timesteps, values, doses, masks = [], [], [], []

        for row, bucket in enumerate(buckets):
            offsets[row + 1] = offsets[row] + len(bucket)
            for (timestep, value, dose, mask) in bucket:
                timesteps.append(timestep)
                if isinstance(value, str):
                    value = np.array(
                        [text_table.setdefault(value, len(text_table))],
                        dtype=np.int32
                    )
                values.append(value)
                if n_slots > 1:
                    doses.append(dose)
                    masks.append(mask)

        entry = {
            'offsets': offsets,
            'timesteps': np.array(timesteps, dtype=np.int32),
        }
        stacked = (np.stack(values, axis=0) if values
                   else np.zeros((0, n_slots), dtype=np.int32))
        entry['values'] = (
            stacked.reshape(-1).astype(np.int32) if n_slots == 1
            else stacked.astype(np.int32)
        )
        if n_slots > 1:
            entry['doses'] = (
                np.stack(doses, axis=0) if doses
                else np.zeros((0, n_slots), dtype=np.float32)
            ).astype(np.float32)
            entry['masks'] = (
                np.stack(masks, axis=0) if masks
                else np.zeros((0, n_slots), dtype=np.float32)
            ).astype(np.float32)
        finalized.append(entry)

    del arrays['_lookup_entries']

    strings = [None] * len(text_table)
    for value, index in text_table.items():
        strings[index] = value
    return finalized, strings


def collate_tensorized(batch: MixedDataset) -> MixedTensorDataset:
    """Collate function for MixedDataset.

    Takes a list of episode dicts and stacks them into the
    MixedTensorDataset format expected by the model. Pre-computed text
    embeddings are stacked into a single tensor at
    batch['val_data']['text']['embedded_values'] with shape
    [batch_size, max_ts_len, n_text_feats, embed_dim].

    Section 4.2 removes the history region, so the history-masking
    arguments this took are gone with it.

    Args:
        batch: List of episode dicts from MixedDataset.__getitem__.
    """

    # Stack simple tensors directly
    val_times = torch.stack([b['val_times'] for b in batch], dim=0)
    val_masks = torch.stack([b['val_masks'] for b in batch], dim=0)
    event_times = torch.stack([b['event_times'] for b in batch], dim=0)
    event_masks = torch.stack([b['event_masks'] for b in batch], dim=0)
    static_data = torch.stack([b['static_data'] for b in batch], dim=0)

    # Stack indicator tensors
    val_numeric_ind = torch.stack([b['val_numeric_indicators'] for b in batch], dim=0)
    val_categorical_ind = torch.stack([b['val_categorical_indicators'] for b in batch], dim=0)
    val_ordinal_ind = torch.stack([b['val_ordinal_indicators'] for b in batch], dim=0)
    val_text_ind = torch.stack([b['val_text_indicators'] for b in batch], dim=0)
    event_ind = torch.stack([b['event_indicators'] for b in batch], dim=0)

    # Stack per-feature value tensors
    n_numeric_feats = len(batch[0]['val_numeric_values'])
    n_categorical_feats = len(batch[0]['val_categorical_values'])
    n_ordinal_feats = len(batch[0]['val_ordinal_values'])
    n_text_feats = len(batch[0]['val_text_embeddings'])

    val_numeric_values = [
        torch.stack([b['val_numeric_values'][f] for b in batch], dim=0)
        for f in range(n_numeric_feats)
    ]
    val_categorical_values = [
        torch.stack([b['val_categorical_values'][f] for b in batch], dim=0)
        for f in range(n_categorical_feats)
    ]
    val_ordinal_values = [
        torch.stack([b['val_ordinal_values'][f] for b in batch], dim=0)
        for f in range(n_ordinal_feats)
    ]

    # Stack pre-computed text embeddings into [batch, max_ts, n_text_feats, embed_dim]
    # Each b['val_text_embeddings'][f] has shape [max_ts_len, embed_dim]
    # First stack features: [max_ts_len, n_text_feats, embed_dim] per episode
    # Then stack episodes: [batch_size, max_ts_len, n_text_feats, embed_dim]
    if n_text_feats > 0:
        val_text_embeddings = torch.stack([
            torch.stack(b['val_text_embeddings'], dim=1)  # [max_ts, n_feats, embed_dim]
            for b in batch
        ], dim=0)  # [batch, max_ts, n_feats, embed_dim]
    else:
        val_text_embeddings = None

    # Stack targets
    time_to_event = torch.stack(
        [b['time_to_event'] for b in batch], dim=0
    ).unsqueeze(-1)
    event_type = torch.stack([b['event_type'] for b in batch], dim=0)

    # Build the MixedTensorDataset structure expected by the model
    result = {
        'val_data': {
            'numeric': {
                'indicators': val_numeric_ind,
                'values': val_numeric_values,
            },
            'categorical': {
                'indicators': val_categorical_ind,
                'values': val_categorical_values,
            },
            'ordinal': {
                'indicators': val_ordinal_ind,
                'values': val_ordinal_values,
            },
            'times': val_times,
            'masks': val_masks,
        },
        'event_data': {
            'indicators': event_ind,
            'times': event_times,
            'masks': event_masks,
        },
        'static_data': static_data,
        'targets': {
            'time_to_event': time_to_event,
            'event_type': event_type,
        },
    }

    if val_text_embeddings is not None:
        result['val_data']['text'] = {
            'indicators': val_text_ind,
            'embedded_values': val_text_embeddings,
        }

    # Identifiers for mapping model outputs (e.g. XAI scores) back to the
    # on-disk .npy rows. 'idx' is the array/row index; 'episode_id' is the
    # patient episode ID (present only when the dataset was built with IDs).
    result['idx'] = torch.tensor([b['idx'] for b in batch], dtype=torch.long)
    if 'episode_id' in batch[0]:
        result['episode_id'] = torch.tensor(
            [b['episode_id'] for b in batch], dtype=torch.long
        )

    return result


def save_extracted(
    arrays: Dict[str, np.ndarray],
    lookup_csr: List[Dict[str, np.ndarray]],
    dims: TensorDimensions,
    metadata: dict,
    base_path: str
) -> None:
    """
    Write section 4.4's on-disk contract as memory-mappable .npy files.

    Two things the table asks for that are easy to miss: no
    ``static_data.npy`` when ``STATIC_FEATS`` is empty, since the array
    would be ``(n, 0)``; and the lookup family's dense indicator array is
    split by type, so ``text`` and ``drug`` keep distinct filenames for
    the webapp even though one code path produced them.
    """
    os.makedirs(base_path, exist_ok=True)

    def save_array(name: str, arr: np.ndarray):
        np.save(os.path.join(base_path, f'{name}.npy'), arr)

    save_array('val_times', arrays['val_times'])
    save_array('val_masks', arrays['val_masks'])
    save_array('val_numeric_indicators', arrays['val_numeric_indicators'])
    save_array('val_categorical_indicators',
               arrays['val_categorical_indicators'])
    save_array('val_ordinal_indicators', arrays['val_ordinal_indicators'])
    save_array('event_indicators', arrays['event_indicators'])
    save_array('event_times', arrays['event_times'])
    save_array('event_masks', arrays['event_masks'])
    save_array('index_times', arrays['index_times'])
    save_array('time_to_event', arrays['time_to_event'])
    save_array('event_type', arrays['event_type'])

    if dims.static_total_dim > 0:
        save_array('static_data', arrays['static_data'])

    for i, arr in enumerate(arrays['val_numeric_values']):
        save_array(f'val_numeric_values_{i}', arr)
    for i, arr in enumerate(arrays['val_categorical_values']):
        save_array(f'val_categorical_values_{i}', arr)
    for i, arr in enumerate(arrays['val_ordinal_values']):
        save_array(f'val_ordinal_values_{i}', arr)

    # One dense indicator array per lookup *type*, and CSR files named by
    # type with an index within that type (section 4.4).
    per_type: Counter = Counter()
    for f, feat_type in enumerate(metadata['lookup_feat_types']):
        j = per_type[feat_type]
        per_type[feat_type] += 1
        for key, arr in lookup_csr[f].items():
            save_array(f'{feat_type}_{key}_{j}', arr)
    for feat_type in per_type:
        columns = [
            f for f, t in enumerate(metadata['lookup_feat_types'])
            if t == feat_type
        ]
        save_array(
            f'val_{feat_type}_indicators',
            arrays['val_lookup_indicators'][:, :, columns]
        )

    with open(os.path.join(base_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Saved extracted dataset to {base_path}/")


def load_dataset(base_path: str) -> MixedDataset:
    """
    Load tensorized dataset with memory-mapped arrays.

    NOTE: this still reads the pre-C1 per-episode text layout and the
    mortality / length-of-stay / phenotype targets. It is C2's to bring on
    to section 4.4's contract, together with ``MixedDataset.__getitem__``,
    which is the only consumer of what it returns.
    """
    def load_mmap(name: str) -> np.ndarray:
        return np.load(os.path.join(base_path, f'{name}.npy'), mmap_mode='r')

    with open(os.path.join(base_path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # Row idx -> patient episode ID (sibling of the dataset dir). Absent for
    # datasets built before IDs were threaded through; None keeps it optional.
    episode_ids = None
    ids_path = f'{base_path}_ids.pkl'
    if os.path.exists(ids_path):
        with open(ids_path, 'rb') as f:
            episode_ids = pickle.load(f)

    n_num = metadata['n_numeric_feats']
    n_cat = metadata['n_categorical_feats']
    n_ord = metadata.get('n_ordinal_feats', 0)
    n_txt = metadata['n_text_feats']
    text_embed_dim = metadata.get('text_embed_dim', 0)

    # Load pre-computed embeddings if available (backward-compatible)
    val_text_embeddings = []
    if text_embed_dim > 0:
        for i in range(n_txt):
            embed_path = os.path.join(base_path, f'val_text_embeddings_{i}.npy')
            if os.path.exists(embed_path):
                val_text_embeddings.append(
                    np.load(embed_path, mmap_mode='r')
                )

    return MixedDataset(
        val_numeric_indicators=load_mmap('val_numeric_indicators'),
        val_numeric_values=[load_mmap(f'val_numeric_values_{i}') for i in range(n_num)],
        val_categorical_indicators=load_mmap('val_categorical_indicators'),
        val_categorical_values=[load_mmap(f'val_categorical_values_{i}') for i in range(n_cat)],
        val_ordinal_indicators=load_mmap('val_ordinal_indicators') if n_ord > 0 else np.empty((0, 0, 0), dtype=np.float32),
        val_ordinal_values=[load_mmap(f'val_ordinal_values_{i}') for i in range(n_ord)],
        val_text_indicators=load_mmap('val_text_indicators'),
        val_times=load_mmap('val_times'),
        val_masks=load_mmap('val_masks'),
        val_text_offsets=[load_mmap(f'val_text_offsets_{i}') for i in range(n_txt)],
        val_text_values=[load_mmap(f'val_text_values_{i}') for i in range(n_txt)],
        val_text_masks=[load_mmap(f'val_text_masks_{i}') for i in range(n_txt)],
        val_text_timesteps=[load_mmap(f'val_text_timesteps_{i}') for i in range(n_txt)],
        val_text_embeddings=val_text_embeddings,
        text_embed_dim=text_embed_dim,
        event_indicators=load_mmap('event_indicators'),
        event_times=load_mmap('event_times'),
        event_masks=load_mmap('event_masks'),
        static_data=load_mmap('static_data'),
        mortality=load_mmap('mortality'),
        length_of_stay=load_mmap('length_of_stay'),
        phenotype=load_mmap('phenotype'),
        max_ts_len=metadata['max_ts_len'],
        text_token_len=metadata['text_token_len'],
        episode_ids=episode_ids,
    )


def standardize_feats(
    arrays: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    dims: TensorDimensions,
    rows: np.ndarray,
    save_path: str
) -> None:
    """Compute and save one fold's numeric standardization statistics.

    **The values are no longer scaled in place** (section 5). Extraction
    now runs once for the cohort while standardization is per fold, so a
    single array cannot carry one fold's scaling; ``__getitem__`` applies
    the statistics at load time instead (C2).

    Statistics are computed over the observed values (indicator == 1.0) of
    the given rows only, which must be that fold's *training* rows --
    computing them over the whole cohort would leak val and test.

    Args:
        arrays: The pre-allocated output arrays.
        dims: Tensor dimensions, for ``n_numeric_feats``.
        rows: int64 row indices of the fold's training episodes.
        save_path: ``summary_statistics_fold{i}.npz`` to write.

    Returns:
        None. Nothing in ``arrays`` is modified.
    """

    n_feats = dims.n_numeric_feats
    means = np.zeros(n_feats, dtype=np.float32)
    p5 = np.zeros(n_feats, dtype=np.float32)
    p95 = np.zeros(n_feats, dtype=np.float32)

    # Indexed one feature at a time: ``indicators[rows]`` would
    # materialize a (len(rows), T, 94) copy, which at cohort scale is
    # gigabytes for a statistic that needs one column at a time.
    indicators = arrays['val_numeric_indicators']

    for f in range(n_feats):
        values = arrays['val_numeric_values'][f][rows]
        mask = indicators[rows, :, f] == 1.0

        if mask.any():
            observed = values[mask]
            means[f] = observed.mean()
            norms = np.linalg.norm(observed, ord=2, axis=-1)
            p5[f] = np.percentile(norms, 5)
            p95[f] = np.percentile(norms, 95)

    np.savez(save_path, means=means, p5=p5, p95=p95)


def get_text_counts_from_dataset_vectorized(dataset) -> np.ndarray:
    """
    Compute total text entry count per episode (vectorized version).

    Args:
        dataset: MixedDataset instance with sparse text storage

    Returns:
        Array of shape (n_episodes,) with total text entries per episode
    """
    n_episodes = dataset.n_episodes
    text_counts = np.zeros(n_episodes, dtype=np.int32)

    for f in range(dataset.n_text_feats):
        offsets = np.asarray(dataset.val_text_offsets[f])
        text_counts += (offsets[1:] - offsets[:-1]).astype(np.int32)

    return text_counts


def extract_data(
    reader: EHRDataReader,
    output_dir: str,
    var_properties_path: str,
    max_episode_len_steps: int,
    clinvec_path: Optional[str] = None,
    min_episode_len_steps: int = 1,
    n_workers: int = 1,
    fold_train_rows: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Extract the cohort's episodes into section 4.4's on-disk contract.

    Runs **once for the cohort**, not once per fold and not once per
    partition: the row order is ``labels.csv``'s, and folds are row
    indices into the result (section 3). There is no ``suffix`` argument
    and no per-fold standardization pass over the values.

    The pipeline is two passes:
        1. **Extraction**, parallel by patient (section 4.1) -- one CSV
           read and one typing pass serve all of that patient's episodes.
        2. **Insertion**, in ``labels.csv`` order, which is where text
           strings are interned into their table row indices.

    Args:
        reader: A patient-indexed EHRDataReader.
        output_dir: Where ``extracted/`` and its siblings are written.
        var_properties_path: Path to variable_properties.yaml.
        max_episode_len_steps: Timesteps per episode, T of section 4.4.
        clinvec_path: ``ClinVec_atc.csv``; required when a drug feature is
            configured, for the pad index of section 4.4.
        min_episode_len_steps: Episodes below this are *reported*, never
            dropped -- the fold row indices fix the row count.
        n_workers: Parallel worker processes.
        fold_train_rows: ``{fold_name: row indices}`` for the per-fold
            standardization statistics C2 loads.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_patients = len(reader)
    n_episodes = reader.n_episodes
    lookup_feats = list(reader.text_feats) + list(reader.drug_feats)

    print(f"Processing {n_episodes} episodes over {n_patients} patients "
          f"using {n_workers} worker(s)...")
    sys.stdout.flush()

    dims = _get_tensor_dimensions(
        var_properties_path=var_properties_path,
        valued_feats=reader.valued_feats,
        event_feats=reader.event_feats,
        lookup_feats=lookup_feats,
        static_feats=reader.static_feats,
        max_ts_len=max_episode_len_steps,
        n_episodes=n_episodes,
        clinvec_path=clinvec_path
    )
    dims_dict = dims.__dict__.copy()

    with open(var_properties_path, 'r') as f:
        var_properties = yaml.safe_load(f)
    numeric_feats, categorical_feats, ordinal_feats, lookup = (
        _bucket_valued_feats(reader.valued_feats, lookup_feats,
                             var_properties)
    )

    process_fn = partial(
        _process_single_patient,
        reader=reader,
        max_episode_len_steps=max_episode_len_steps
    )

    print("Allocating output arrays...")
    sys.stdout.flush()
    arrays = _allocate_output_arrays(dims)

    print("Extracting patients and inserting their episodes...")
    sys.stdout.flush()

    # Episodes are inserted the moment their patient comes back, in
    # whatever order that is. Only the CSR entries are held, bucketed by
    # row, and the row order is restored when they are concatenated --
    # so the text table's key order is still canonical, while the dense
    # per-episode arrays are freed as they are consumed.
    cat_misses = np.zeros(dims.n_categorical_feats, dtype=np.int64)
    ord_misses = np.zeros(dims.n_ordinal_feats, dtype=np.int64)
    inserted = np.zeros(n_episodes, dtype=bool)
    short_episodes = 0
    errors = []

    with mp.Pool(
        processes=n_workers,
        initializer=_init_tensorized_worker,
        initargs=(var_properties_path, reader.valued_feats,
                  reader.event_feats, lookup_feats, reader.static_feats,
                  dims_dict)
    ) as pool:
        for episodes, patient_cat, patient_ord, error in tqdm(
            pool.imap_unordered(process_fn, range(n_patients), chunksize=1),
            total=n_patients,
            desc="Extracting"
        ):
            if error is not None:
                errors.append(error)
                continue
            cat_misses += patient_cat
            ord_misses += patient_ord
            for episode in episodes:
                _insert_episode(arrays, dims, episode)
                inserted[episode.row] = True
                if episode.val_len < min_episode_len_steps:
                    short_episodes += 1

    if not inserted.all():
        for error in errors[:10]:
            print(f"  {error}")
        raise RuntimeError(
            f"{int((~inserted).sum())} of {n_episodes} episode(s) produced "
            f"no data ({len(errors)} patient(s) failed). Section 3's fold "
            f"row indices are positions in labels.csv, so extraction "
            f"cannot skip a row."
        )

    lookup_csr, strings = _finalize_sparse_lookup(arrays, dims)
    episode_ids = list(
        zip(reader.labels['PATID'].tolist(),
            reader.labels['STAY_INDEX'].tolist())
    )
    gc.collect()

    metadata = {
        'max_ts_len': dims.max_ts_len,
        'n_numeric_feats': dims.n_numeric_feats,
        'n_categorical_feats': dims.n_categorical_feats,
        'n_ordinal_feats': dims.n_ordinal_feats,
        'n_lookup_feats': dims.n_lookup_feats,
        'n_event_feats': dims.n_event_feats,
        'numeric_feats': numeric_feats,
        'categorical_feats': categorical_feats,
        'ordinal_feats': ordinal_feats,
        'lookup_feats': lookup,
        'event_feats': list(reader.event_feats),
        'numeric_feat_dims': dims.numeric_feat_dims,
        'categorical_feat_dims': dims.categorical_feat_dims,
        'ordinal_feat_dims': dims.ordinal_feat_dims,
        'lookup_slot_dims': dims.lookup_slot_dims,
        'lookup_table_dims': dims.lookup_table_dims,
        'lookup_pad_indices': dims.lookup_pad_indices,
        'lookup_feat_types': [
            var_properties[f]['type'] for f in lookup
        ],
        'static_feat_dims': dims.static_feat_dims,
        'static_total_dim': dims.static_total_dim,
        # Section 4.4 also asks for the resolved pad_token_id; C4 adds it,
        # since that is where the tokenizer is loaded.
        'LLM_NAME': LLM_NAME,
        'TOKENIZER_PAD_TOKEN': TOKENIZER_PAD_TOKEN,
        'MAX_TOKEN_LENGTH': MAX_TOKEN_LENGTH,
    }

    output_path = os.path.join(output_dir, 'extracted')
    save_extracted(arrays, lookup_csr, dims, metadata, output_path)

    with open(os.path.join(output_path, 'episode_ids.pkl'), 'wb') as f:
        pickle.dump(episode_ids, f)

    # The unique-string table's key order. C4 embeds this list in this
    # order; ``text_values`` are positions in it.
    with open(os.path.join(output_path, 'text_strings.pkl'), 'wb') as f:
        pickle.dump(strings, f)

    for fold_name, rows in (fold_train_rows or {}).items():
        stats_path = os.path.join(
            output_path, f'summary_statistics_{fold_name}.npz'
        )
        standardize_feats(arrays, dims, np.asarray(rows), stats_path)
        print(f"Wrote {stats_path}")

    _report_extraction(
        n_episodes, len(strings), short_episodes,
        min_episode_len_steps, categorical_feats, cat_misses,
        ordinal_feats, ord_misses
    )


def _insert_episode(
    arrays: Dict[str, np.ndarray],
    dims: TensorDimensions,
    episode: EpisodeData
) -> None:
    """Place one episode into the pre-allocated arrays, right-aligned.

    Section 4.2 puts ``t = 0`` -- the index time, and the last record --
    in the final column of the row, so an episode of length L occupies
    ``[T - L, T)`` and the left of the row stays zero padding.
    """
    row = episode.row
    ts = dims.max_ts_len

    val_len = min(episode.val_len, ts)
    if val_len > 0:
        start = ts - val_len
        arrays['val_times'][row, start:] = episode.val_times[-val_len:]
        arrays['val_masks'][row, start:] = 1.0
        arrays['val_numeric_indicators'][row, start:, :] = (
            episode.val_numeric_indicators[-val_len:]
        )
        for f, values in enumerate(episode.val_numeric_values):
            arrays['val_numeric_values'][f][row, start:, :] = (
                values[-val_len:]
            )
        arrays['val_categorical_indicators'][row, start:, :] = (
            episode.val_categorical_indicators[-val_len:]
        )
        for f, values in enumerate(episode.val_categorical_values):
            arrays['val_categorical_values'][f][row, start:] = (
                values[-val_len:]
            )
        arrays['val_ordinal_indicators'][row, start:, :] = (
            episode.val_ordinal_indicators[-val_len:]
        )
        for f, values in enumerate(episode.val_ordinal_values):
            arrays['val_ordinal_values'][f][row, start:] = values[-val_len:]
        arrays['val_lookup_indicators'][row, start:, :] = (
            episode.val_lookup_indicators[-val_len:]
        )
    else:
        start = ts

    # Text values are still strings here: they are interned in row
    # order by ``_finalize_sparse_lookup``, which is the only place that
    # sees the rows in canonical order.
    for f, entries in enumerate(episode.val_lookup_sparse):
        bucket = arrays['_lookup_entries'][f][row]
        for (t, values, doses, masks) in entries:
            timestep = start + t
            if 0 <= timestep < ts:
                bucket.append((timestep, values, doses, masks))

    event_len = min(episode.event_len, ts)
    if event_len > 0:
        e_start = ts - event_len
        arrays['event_times'][row, e_start:] = (
            episode.event_times[-event_len:]
        )
        arrays['event_masks'][row, e_start:] = 1.0
        arrays['event_indicators'][row, e_start:, :] = (
            episode.event_indicators[-event_len:]
        )

    if dims.static_total_dim > 0:
        arrays['static_data'][row, :] = episode.static_data
    arrays['time_to_event'][row] = episode.time_to_event
    arrays['event_type'][row] = episode.event_type
    arrays['index_times'][row] = episode.index_time


def _report_extraction(
    n_episodes: int,
    n_strings: int,
    short_episodes: int,
    min_episode_len_steps: int,
    categorical_feats: List[str],
    cat_misses: np.ndarray,
    ordinal_feats: List[str],
    ord_misses: np.ndarray
) -> None:
    """Print the per-feature miss report section 4.3 asks for.

    A miss is otherwise invisible: the timestep keeps indicator 1 and
    stores -1, so nothing raises. Section A.2's enforcement pass
    guarantees every LAB result is one of its declared levels, so a
    nonzero count on a LAB feature is a bug by construction rather than
    data drift. Do not raise on one: ``INST``, ``CMG`` and ``SCU`` are
    hand-curated against a single extract, and a new institution name
    should not kill a multi-hour run.
    """
    print(f"\nExtracted {n_episodes} episode(s); "
          f"{n_strings} unique text string(s).")

    if short_episodes:
        print(f"\n{short_episodes} episode(s) have fewer than "
              f"{min_episode_len_steps} timestep(s). They are written "
              f"anyway -- the fold row indices are positions in "
              f"labels.csv -- but invariant 4 says every INDEX_TIME "
              f"names a timeseries.csv row, so this should be zero.")

    misses = [
        (feat, int(count))
        for feats, counts in ((categorical_feats, cat_misses),
                              (ordinal_feats, ord_misses))
        for feat, count in zip(feats, counts) if count
    ]
    if not misses:
        print("\nNo out-of-domain categorical or ordinal values.")
        return
    total = sum(count for _, count in misses)
    print(f"\nOut-of-domain values, stored as -1 ({total} over "
          f"{len(misses)} feature(s)):")
    for feat, count in sorted(misses, key=lambda pair: -pair[1]):
        print(f"  {feat:<24} {count:>10}")
    print("  A value not in the feature's category_map. Not an error for "
          "INST, CMG or SCU,\n  which are hand-curated; on a LAB feature "
          "it is a bug (section 4.3).")


def prepare_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    balance_text: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> List[DataLoader]:
    """Prepare training, (validation), and test DataLoaders for MixedDataset.

    This function creates PyTorch DataLoader instances for `MixedDataset` objects prepared by
    `extract_data()`. The dataset uses memory-mapped numpy arrays for efficient multi-worker
    access with minimal memory overhead. Workers share read-only memory-mapped arrays rather
    than duplicating data in each worker process' memory space.

    NOTE: like ``load_dataset``, this still assumes the pre-C1 layout of
    one directory per fold partition. C2 brings it on to the cohort-wide
    arrays and the fold row indices of section 3.

    Args:
        data_dir (str): Directory containing 'train/', 'val/', and 'test/' subdirectories. Each
            subdirectory should be the output of `extract_data()`.
        batch_size (int): Number of samples per batch (per GPU in distributed settings).
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfers.
            Defaults to True. Only effective if num_workers > 0.
        prefetch_factor (int, optional): Number of batches to prefetch per worker. Defaults to 2.
            Higher values increase memory usage but can improve throughput if batch processing by
            the model is slower than data loading. Only effective if num_workers > 0.
        balance_text (bool, optional): If True and running distributed (world_size > 1), use
            TextBalancedDistributedSampler to balance text density across ranks for all partitions.
            Defaults to False.
        world_size (int, optional): Number of distributed processes. Required if balance_text=True.
        rank (int, optional): Current process rank. Required if balance_text=True.

    Returns:
        List[DataLoader]: List of DataLoaders in order: [train_loader, val_loader (if available),
            test_loader]. If validation data are not found, only [train_loader, test_loader] is
            returned.

    Raises:
        FileNotFoundError: If 'train/' or 'test/' directories are not found in `data_dir`.
        ValueError: If balance_text=True but world_size or rank is not provided.
    """
    if balance_text and (world_size is None or rank is None):
        raise ValueError("world_size and rank are required when balance_text=True")

    dataloaders = []

    for partition in ['train', 'val', 'test']:
        dataset_path = os.path.join(data_dir, partition)

        if not os.path.exists(dataset_path):
            if partition == 'val':
                continue
            else:
                raise FileNotFoundError(f'{partition}/ not found in {data_dir}')

        dataset = load_dataset(dataset_path)

        # Determine sampler and shuffle behavior
        sampler = None
        shuffle = (partition == 'train')

        # Only add balanced sampler if explicitly requested AND distributed
        if balance_text and world_size is not None and world_size > 1:
            text_counts = get_text_counts_from_dataset_vectorized(dataset)
            sampler = TextBalancedDistributedSampler(
                dataset=dataset,
                text_counts=text_counts,
                batch_size=batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,  # True for train, False for val/test
                drop_last=False
            )
            shuffle = False  # Sampler handles shuffling

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate_tensorized,
            num_workers=num_workers,
            pin_memory=pin_memory if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            multiprocessing_context='spawn' if num_workers > 0 else None
        )

        dataloaders.append(loader)

    return dataloaders
