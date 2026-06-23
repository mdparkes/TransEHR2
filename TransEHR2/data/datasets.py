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

    * *max_ts_len* (int): Maximum timeseries length across all episodes.

    * *text_token_len* (List[int]): List of token sequence lengths for each text feature.

    The __getitem__ method returns a dictionary with torch tensors, reconstructing dense text
    embedding arrays on-the-fly from the sparse storage format.
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
    ):
        self.n_episodes = val_times.shape[0]
        self.max_ts_len = max_ts_len
        self.n_text_feats = len(text_token_len)
        self.text_token_len = text_token_len
        self.text_embed_dim = text_embed_dim

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

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> Dict:
        """
        Return episode as torch tensors, reconstructing dense text embeddings on-the-fly.
        """
        # Reconstruct dense text embeddings from sparse storage
        text_embeddings_dense = []

        for f in range(self.n_text_feats):
            dense_embeds = np.zeros(
                (self.max_ts_len, self.text_embed_dim), dtype=np.float32
            )

            start = int(self.val_text_offsets[f][idx])
            end = int(self.val_text_offsets[f][idx + 1])

            if end > start and len(self.val_text_embeddings) > f:
                timesteps = self.val_text_timesteps[f][start:end]
                embeddings = self.val_text_embeddings[f][start:end]

                for i in range(end - start):
                    ts = int(timesteps[i])
                    dense_embeds[ts] = embeddings[i]

            text_embeddings_dense.append(torch.from_numpy(dense_embeds))

        # Return tensors (copy from mmap)
        return {
            'val_numeric_indicators': torch.from_numpy(self.val_numeric_indicators[idx].copy()),
            'val_numeric_values': [torch.from_numpy(v[idx].copy()) for v in self.val_numeric_values],
            'val_categorical_indicators': torch.from_numpy(self.val_categorical_indicators[idx].copy()),
            'val_categorical_values': [torch.from_numpy(v[idx].copy()) for v in self.val_categorical_values],
            'val_ordinal_indicators': torch.from_numpy(self.val_ordinal_indicators[idx].copy()) if self.val_ordinal_indicators.size > 0 else torch.empty(0),
            'val_ordinal_values': [torch.from_numpy(v[idx].copy()) for v in self.val_ordinal_values],
            'val_multilabel_indicators': torch.from_numpy(self.val_multilabel_indicators[idx].copy()) if self.val_multilabel_indicators.size > 0 else torch.empty(0),
            'val_multilabel_values': [torch.from_numpy(v[idx].copy()) for v in self.val_multilabel_values],
            'val_text_indicators': torch.from_numpy(self.val_text_indicators[idx].copy()),
            'val_text_embeddings': text_embeddings_dense,
            'val_times': torch.from_numpy(self.val_times[idx].copy()),
            'val_masks': torch.from_numpy(self.val_masks[idx].copy()),
            'event_indicators': torch.from_numpy(self.event_indicators[idx].copy()),
            'event_times': torch.from_numpy(self.event_times[idx].copy()),
            'event_masks': torch.from_numpy(self.event_masks[idx].copy()),
            'static_data': torch.from_numpy(self.static_data[idx].copy()),
            'mortality': torch.tensor(float(self.mortality[idx]), dtype=torch.float32),
            'length_of_stay': torch.tensor(float(self.length_of_stay[idx]), dtype=torch.float32),
            'phenotype': torch.from_numpy(self.phenotype[idx].copy()),
        }
