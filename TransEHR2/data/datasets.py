import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Dict, List, Optional


class MixedDataset(Dataset):
    """A dataset for input to a TransEHR2 model.

    The dataset stores patient-episode data using memory-mapped numpy arrays for efficient
    multi-worker DataLoader compatibility. Conversion to torch tensors happens only in
    __getitem__ for the requested episode.

    The dataset contains the following data structures:

    * *val_numeric_indicators* (np.ndarray): Array of shape (n_episodes, max_ts_len, n_numeric_features)
      indicating whether each numeric feature was recorded at each timestep (1) or not (0).

    * *val_numeric_values* (List[np.ndarray]): List of n_numeric_features arrays, each of shape
      (n_episodes, max_ts_len, feature_dim), containing the actual numeric values recorded.
      Arrays default to zeros if the feature was not recorded at that timestep.

    * *val_categorical_indicators* (np.ndarray): Array of shape (n_episodes, max_ts_len, n_categorical_features)
      indicating whether each categorical feature was recorded at each timestep.

    * *val_categorical_values* (List[np.ndarray]): List of n_categorical_features arrays, each of shape
      (n_episodes, max_ts_len, n_classes), containing one-hot encoded categorical values.

    * *val_text_indicators* (np.ndarray): Array of shape (n_episodes, max_ts_len, n_text_features)
      indicating whether each text feature was recorded at each timestep.

    * *val_times* (np.ndarray): Array of shape (n_episodes, max_ts_len) containing the times
      at which values were recorded. Padded with zeros up to max_ts_len.

    * *val_masks* (np.ndarray): Array of shape (n_episodes, max_ts_len) indicating whether each
      timestep is part of the episode (1) or length padding (0).

    * *val_text_offsets* (List[np.ndarray]): List of n_text_features arrays, each of shape
      (n_episodes + 1,), containing CSR-style offsets into the sparse text storage.

    * *val_text_values* (List[np.ndarray]): List of n_text_features arrays containing sparse
      token ID sequences. Shape is (n_non_empty_entries, token_len) for each feature.
      Retained for XAI token-level attribution.

    * *val_text_masks* (List[np.ndarray]): List of n_text_features arrays containing sparse
      attention masks for the token sequences, matching val_text_values shapes.
      Retained for XAI token-level attribution.

    * *val_text_timesteps* (List[np.ndarray]): List of n_text_features arrays containing the
      timestep indices for each sparse text entry.

    * *val_text_embeddings* (List[np.ndarray]): List of n_text_features arrays containing sparse
      pre-computed LLM embedding vectors. Shape is (n_non_empty_entries, embed_dim) for each
      feature. These are mean-pooled LLM outputs used as model input features.

    * *text_embed_dim* (int): Dimensionality of the pre-computed text embeddings.

    * *event_indicators* (np.ndarray): Array of shape (n_episodes, max_ts_len, n_event_types)
      indicating whether each event type occurred at each timestep.

    * *event_times* (np.ndarray): Array of shape (n_episodes, max_ts_len) containing event times.

    * *event_masks* (np.ndarray): Array of shape (n_episodes, max_ts_len) indicating valid
      event timesteps vs padding.

    * *static_data* (np.ndarray): Array of shape (n_episodes, n_static_features) containing
      time-invariant patient features.

    * *mortality* (np.ndarray): Array of shape (n_episodes,) containing binary mortality labels.

    * *length_of_stay* (np.ndarray): Array of shape (n_episodes,) containing length of stay values.

    * *phenotype* (np.ndarray): Array of shape (n_episodes, n_phenotypes) containing multi-label
      phenotype indicators.

    * *max_ts_len* (int): Maximum timeseries length across all episodes, as extracted. Equals
      `max_history_len_steps + max_episode_len_steps` from the extraction configuration.

    * *max_history_len_steps* (int): Number of leading timestep indices reserved for
      pre-admission historical records at extraction time. Episode (in-stay) data always
      begins at index `max_history_len_steps`.

    * *text_token_len* (List[int]): List of token sequence lengths for each text feature.

    **Runtime sequence-length cropping**

    The extracted layout is suffix-truncatable on the history side and prefix-truncatable on the
    episode side, so shorter sequence limits can be applied at load time without re-extracting:

        index:  0 .......... max_history_len_steps .......... max_ts_len
                [ pad | history (right-justified) ][ episode (left-justified) ]

    `filter_timeseries_records()` retains the *most recent* `min(n_historic, H)` historical
    records and the *earliest* `min(n_episode, E)` in-stay records. Cropping the stored window to
    `[max_history_len_steps - H_new, max_history_len_steps + E_new)` therefore yields exactly the
    arrays that an extraction with `H_new`/`E_new` would have produced, provided
    `H_new <= max_history_len_steps` and `E_new <= max_episode_len_steps`. Episode inclusion
    criteria depend only on the in-stay portion before truncation, so the set of episodes is
    unaffected.

    Note that numeric standardization statistics are computed at extraction time over the full
    extracted window, so cropped data remains normalized against all available timesteps. This is
    intentional: it holds normalization constant across a sequence-length sweep.

    * *history_len_steps* (Optional[int]): Runtime cap on historical timesteps. None uses all
      extracted history.

    * *episode_len_steps* (Optional[int]): Runtime cap on in-stay timesteps. None uses all
      extracted episode steps.

    * *ts_len* (int): Width of the timestep axis actually returned by __getitem__, i.e.
      `min(history_len_steps, max_history_len_steps) + min(episode_len_steps, max_episode_len_steps)`.

    The __getitem__ method returns a dictionary with torch tensors, cropped to the configured
    window and reconstructing dense text embedding arrays on-the-fly from the sparse storage
    format.
    """

    def __init__(
        self,
        # All inputs are numpy arrays (potentially memory-mapped)
        val_numeric_indicators: np.ndarray,
        val_numeric_values: List[np.ndarray],
        val_categorical_indicators: np.ndarray,
        val_categorical_values: List[np.ndarray],
        val_ordinal_indicators: np.ndarray,
        val_ordinal_values: List[np.ndarray],
        val_multilabel_indicators: np.ndarray,
        val_multilabel_values: List[np.ndarray],
        val_text_indicators: np.ndarray,
        val_times: np.ndarray,
        val_masks: np.ndarray,
        val_text_offsets: List[np.ndarray],
        val_text_values: List[np.ndarray],
        val_text_masks: List[np.ndarray],
        val_text_timesteps: List[np.ndarray],
        val_text_embeddings: List[np.ndarray],
        text_embed_dim: int,
        event_indicators: np.ndarray,
        event_times: np.ndarray,
        event_masks: np.ndarray,
        static_data: np.ndarray,
        mortality: np.ndarray,
        length_of_stay: np.ndarray,
        phenotype: np.ndarray,
        max_ts_len: int,
        text_token_len: List[int],
        max_history_len_steps: int = 0,
        history_len_steps: Optional[int] = None,
        episode_len_steps: Optional[int] = None,
    ):
        """Initialize an instance.

        Args:
            max_ts_len: Width of the extracted timestep axis
                (`max_history_len_steps + max_episode_len_steps`).
            text_token_len: Token sequence length for each text feature.
            max_history_len_steps: Number of leading timestep indices reserved for historical
                records at extraction time. Defaults to 0 (no history region).
            history_len_steps: Runtime cap on historical timesteps. Must not exceed
                `max_history_len_steps`. None (default) uses all extracted history.
            episode_len_steps: Runtime cap on in-stay timesteps. Must not exceed
                `max_ts_len - max_history_len_steps`. None (default) uses all extracted
                episode steps.

        Raises:
            ValueError: If a requested length is negative or exceeds what was extracted.
        """
        self.n_episodes = val_times.shape[0]
        self.max_ts_len = max_ts_len
        self.max_history_len_steps = max_history_len_steps
        self.max_episode_len_steps = max_ts_len - max_history_len_steps
        self.n_text_feats = len(text_token_len)
        self.text_token_len = text_token_len
        self.text_embed_dim = text_embed_dim

        # Resolve the runtime crop window. Cropping is only valid in the shortening direction:
        # the extracted arrays cannot manufacture timesteps that were never extracted.
        self.history_len_steps = self._resolve_len(
            history_len_steps, self.max_history_len_steps, 'history_len_steps',
            'MAX_HISTORY_LEN_STEPS'
        )
        self.episode_len_steps = self._resolve_len(
            episode_len_steps, self.max_episode_len_steps, 'episode_len_steps',
            'MAX_EPISODE_LEN_STEPS'
        )

        # History is right-justified in [0, max_history_len_steps) and episode data is
        # left-justified from max_history_len_steps, so a single contiguous slice applies both
        # caps at once, for value- and event-associated arrays alike.
        self.ts_start = self.max_history_len_steps - self.history_len_steps
        self.ts_end = self.max_history_len_steps + self.episode_len_steps
        self.ts_len = self.ts_end - self.ts_start
        self.is_cropped = (self.ts_start != 0) or (self.ts_end != self.max_ts_len)

        # Store numpy arrays directly
        self.val_numeric_indicators = val_numeric_indicators
        self.val_numeric_values = val_numeric_values
        self.val_categorical_indicators = val_categorical_indicators
        self.val_categorical_values = val_categorical_values
        self.val_ordinal_indicators = val_ordinal_indicators
        self.val_ordinal_values = val_ordinal_values
        self.val_multilabel_indicators = val_multilabel_indicators
        self.val_multilabel_values = val_multilabel_values
        self.val_text_indicators = val_text_indicators
        self.val_times = val_times
        self.val_masks = val_masks
        self.val_text_offsets = val_text_offsets
        self.val_text_values = val_text_values
        self.val_text_masks = val_text_masks
        self.val_text_timesteps = val_text_timesteps
        self.val_text_embeddings = val_text_embeddings
        self.event_indicators = event_indicators
        self.event_times = event_times
        self.event_masks = event_masks
        self.static_data = static_data
        self.mortality = mortality
        self.length_of_stay = length_of_stay
        self.phenotype = phenotype

    @staticmethod
    def _resolve_len(requested: Optional[int], extracted: int, arg_name: str,
                     config_key: str) -> int:
        """Validate a requested runtime length against what was extracted.

        Args:
            requested: Requested number of timesteps, or None to use all extracted steps.
            extracted: Number of timesteps available in the extracted arrays.
            arg_name: Name of the constructor argument, for error messages.
            config_key: Name of the extraction-time config key, for error messages.

        Returns:
            The number of timesteps to retain.

        Raises:
            ValueError: If `requested` is negative or exceeds `extracted`.
        """
        if requested is None:
            return extracted
        if requested < 0:
            raise ValueError(f'{arg_name} must be non-negative, got {requested}')
        if requested > extracted:
            raise ValueError(
                f'{arg_name}={requested} exceeds the {extracted} timesteps present in the '
                f'extracted data. Sequence limits can only be shortened at load time; to go '
                f'longer, re-extract with a larger {config_key}.'
            )
        return requested

    def __len__(self) -> int:
        return self.n_episodes

    def _empty_indicators(self) -> torch.Tensor:
        """Indicator tensor for a feature type the extraction produced no features for.

        `load_dataset` substitutes a `(0, 0, 0)` array whenever metadata reports zero features
        of a type, so the per-episode slice cannot be taken. The replacement still has to be
        two-dimensional `(timesteps, features)`: `collate_tensorized` stacks these into
        `(batch, timesteps, features)` and `_gen_val_assoc_feat_mask` unpacks exactly three
        dimensions. Returning a bare `torch.empty(0)` collates to `(batch, 0)` and raises
        `not enough values to unpack (expected 3, got 2)` on the first batch.
        """
        return torch.zeros((self.ts_len, 0), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return episode as torch tensors, reconstructing dense text embeddings on-the-fly.
        """
        ts_start, ts_end = self.ts_start, self.ts_end

        # Reconstruct dense text embeddings from sparse storage. Sparse timesteps are absolute
        # indices into the extracted axis, so entries outside the crop window are dropped and
        # the survivors are shifted into the cropped frame.
        text_embeddings_dense = []

        for f in range(self.n_text_feats):
            dense_embeds = np.zeros(
                (self.ts_len, self.text_embed_dim), dtype=np.float32
            )

            start = int(self.val_text_offsets[f][idx])
            end = int(self.val_text_offsets[f][idx + 1])

            if end > start and len(self.val_text_embeddings) > f:
                timesteps = self.val_text_timesteps[f][start:end]
                embeddings = self.val_text_embeddings[f][start:end]

                for i in range(end - start):
                    ts = int(timesteps[i])
                    if ts_start <= ts < ts_end:
                        dense_embeds[ts - ts_start] = embeddings[i]

            text_embeddings_dense.append(torch.from_numpy(dense_embeds))

        # Return tensors (copy from mmap)
        return {
            'val_numeric_indicators': torch.from_numpy(self.val_numeric_indicators[idx, ts_start:ts_end].copy()),
            'val_numeric_values': [torch.from_numpy(v[idx, ts_start:ts_end].copy()) for v in self.val_numeric_values],
            'val_categorical_indicators': torch.from_numpy(self.val_categorical_indicators[idx, ts_start:ts_end].copy()),
            'val_categorical_values': [torch.from_numpy(v[idx, ts_start:ts_end].copy()) for v in self.val_categorical_values],
            'val_ordinal_indicators': torch.from_numpy(self.val_ordinal_indicators[idx, ts_start:ts_end].copy()) if self.val_ordinal_indicators.size > 0 else self._empty_indicators(),
            'val_ordinal_values': [torch.from_numpy(v[idx, ts_start:ts_end].copy()) for v in self.val_ordinal_values],
            'val_multilabel_indicators': torch.from_numpy(self.val_multilabel_indicators[idx, ts_start:ts_end].copy()) if self.val_multilabel_indicators.size > 0 else self._empty_indicators(),
            'val_multilabel_values': [torch.from_numpy(v[idx, ts_start:ts_end].copy()) for v in self.val_multilabel_values],
            'val_text_indicators': torch.from_numpy(self.val_text_indicators[idx, ts_start:ts_end].copy()),
            'val_text_embeddings': text_embeddings_dense,
            'val_times': torch.from_numpy(self.val_times[idx, ts_start:ts_end].copy()),
            'val_masks': torch.from_numpy(self.val_masks[idx, ts_start:ts_end].copy()),
            'event_indicators': torch.from_numpy(self.event_indicators[idx, ts_start:ts_end].copy()),
            'event_times': torch.from_numpy(self.event_times[idx, ts_start:ts_end].copy()),
            'event_masks': torch.from_numpy(self.event_masks[idx, ts_start:ts_end].copy()),
            'static_data': torch.from_numpy(self.static_data[idx].copy()),
            'mortality': torch.tensor(float(self.mortality[idx]), dtype=torch.float32),
            'length_of_stay': torch.tensor(float(self.length_of_stay[idx]), dtype=torch.float32),
            'phenotype': torch.from_numpy(self.phenotype[idx].copy()),
        }
