import gc
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
import torch
import yaml

from collections import namedtuple
from functools import partial
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, Iterator, List, Optional, Tuple, Union

from TransEHR2.constants import HF_API_TOKEN, LLM_NAME, MAX_TOKEN_LENGTH, TOKENIZER_PAD_TOKEN
from TransEHR2.data.custom_types import EpisodeData, MixedTensorDataset, TensorDimensions
from TransEHR2.data.datareaders import MIMICDataReader
from TransEHR2.data.datasets import MixedDataset


# Global variables for multi-process data extraction
_tensorized_processor = None
_tensorized_dims = None


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
        tokenizer_name = 'meta-llama/Llama-3.1-8B'
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
        # Return array of zeros if the text is empty
        if not text or text.strip() == '' or pd.isna(text):
            return {
                'input_ids': np.zeros(self.max_length, dtype=np.int32),
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
    Processes patient episodes directly into numpy arrays for tensor insertion.
    
    This processor outputs flat numpy arrays that can be directly inserted into pre-allocated
    tensors. This eliminates the intermediate representation and reduces memory overhead.
    
    The processor handles:
    - Numeric features: float32 arrays
    - Categorical features: int64 arrays (0 = missing, 1+ = category index)
    - Text features: int64 token arrays + float32 attention masks
    - Event indicators: float32 arrays
    - Static features: concatenated float32 array
    """
    
    def __init__(
        self,
        var_properties_path: str,
        valued_feats: List[str],
        event_feats: List[str],
        text_feats: Optional[List[str]],
        static_feats: List[str],
        dims: TensorDimensions,
        tokenizer: Optional[LLMTextProcessor] = None
    ):
        """
        Initialize the tensorized data processor.
        
        Args:
            var_properties_path: Path to variable_properties.yaml
            valued_feats: List of value-associated feature names
            event_feats: List of event-associated feature names
            text_feats: List of text feature names (may be None)
            static_feats: List of static feature names
            dims: Pre-computed tensor dimensions
            tokenizer: Text tokenizer (required if text_feats is not empty)
        """
        with open(var_properties_path, 'r') as f:
            self.var_properties = yaml.safe_load(f)
        
        # Separate valued_feats by type
        self.numeric_feats = []
        self.categorical_feats = []
        for feat in valued_feats:
            feat_type = self.var_properties[feat]['type']
            if feat_type == 'numeric':
                self.numeric_feats.append(feat)
            elif feat_type == 'categorical':
                self.categorical_feats.append(feat)
        
        self.text_feats = text_feats or []
        self.event_feats = event_feats
        self.static_feats = static_feats
        self.dims = dims
        self.tokenizer = tokenizer
        
        if self.text_feats and tokenizer is None:
            raise ValueError("Tokenizer required when text_feats is not empty")
    
    def process_valued_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray,
               List[np.ndarray], np.ndarray, list]:
        """
        Process value-associated data into flat numpy arrays.

        Text features are returned in sparse format to avoid allocating
        large dense arrays (n_ts, token_len) that are mostly zeros. This
        dramatically reduces multiprocessing IPC overhead since only
        non-empty text entries are pickled and transferred.

        Args:
            data: DataFrame with TimedeltaIndex containing valued features

        Returns:
            Tuple of:
            - times: (n_timesteps,) float32
            - numeric_indicators: (n_timesteps, n_numeric_feats) float32
            - numeric_values: List of (n_timesteps, feat_dim) float32
            - categorical_indicators: (n_timesteps, n_cat_feats) float32
            - categorical_values: List of (n_timesteps, feat_dim) int64
            - text_indicators: (n_timesteps, n_text_feats) float32
            - text_sparse: List of per-feature sparse entries, each a
              list of (timestep, token_ids, attention_mask) tuples
        """
        if data.empty:
            n_ts = 0
        else:
            n_ts = len(data)

        # Pre-allocate output arrays
        times = np.zeros(n_ts, dtype=np.float32)

        # Numeric
        n_num = self.dims.n_numeric_feats
        numeric_indicators = np.zeros((n_ts, n_num), dtype=np.float32)
        numeric_values = [
            np.zeros((n_ts, dim), dtype=np.float32)
            for dim in self.dims.numeric_feat_dims
        ]

        # Categorical
        n_cat = self.dims.n_categorical_feats
        categorical_indicators = np.zeros((n_ts, n_cat), dtype=np.float32)
        categorical_values = [
            np.zeros((n_ts, dim), dtype=np.int64)
            for dim in self.dims.categorical_feat_dims
        ]

        # Text — sparse: only non-empty entries are stored
        n_txt = self.dims.n_text_feats
        text_indicators = np.zeros((n_ts, n_txt), dtype=np.float32)
        text_sparse = [[] for _ in range(n_txt)]

        if data.empty:
            return (times, numeric_indicators, numeric_values,
                    categorical_indicators, categorical_values,
                    text_indicators, text_sparse)

        # Convert column names for namedtuple compatibility
        data = data.copy()
        data.columns = [
            col.replace(' ', '_').replace('-', '_')
            for col in data.columns
        ]

        # Process each timestep
        for t, record in enumerate(
            data.itertuples(index=True, name='Record')
        ):
            # Timestamp in hours
            times[t] = record.Index / np.timedelta64(1, 'h')

            # Numeric features
            for f, feat in enumerate(self.numeric_feats):
                feat_name = feat.replace(' ', '_').replace('-', '_')
                feat_dim = self.dims.numeric_feat_dims[f]
                cols = self._get_feature_columns(feat_name, record)

                if cols:
                    values = [getattr(record, c) for c in cols]
                    if not all(pd.isna(v) for v in values):
                        numeric_indicators[t, f] = 1.0
                        for d, v in enumerate(values):
                            if d < feat_dim and pd.notna(v):
                                numeric_values[f][t, d] = v

            # Categorical features
            for f, feat in enumerate(self.categorical_feats):
                feat_name = feat.replace(' ', '_').replace('-', '_')
                if hasattr(record, feat_name):
                    value = getattr(record, feat_name)
                    if pd.notna(value):
                        categorical_indicators[t, f] = 1.0
                        cat_map = self.var_properties[feat].get(
                            'category_map', {}
                        )
                        if isinstance(value, str):
                            idx_map = {
                                v: int(k)
                                for k, v in cat_map.items()
                            }
                            cat_idx = idx_map.get(value, 0)
                        else:
                            cat_idx = int(value)
                        first_idx = (
                            min(int(k) for k in cat_map.keys())
                            if cat_map else 0
                        )
                        cat_idx = cat_idx - first_idx + 1
                        categorical_values[f][t, 0] = cat_idx

            # Text features — collect sparse entries only
            for f, feat in enumerate(self.text_feats):
                feat_name = feat.replace(' ', '_').replace('-', '_')
                if hasattr(record, feat_name):
                    value = getattr(record, feat_name)
                    if pd.notna(value) and value.strip():
                        text_indicators[t, f] = 1.0
                        tokenized = self.tokenizer.process_text(
                            value
                        )
                        text_sparse[f].append((
                            t,
                            tokenized['input_ids'],
                            tokenized['attention_mask'].astype(
                                np.float32
                            ),
                        ))

        return (times, numeric_indicators, numeric_values,
                categorical_indicators, categorical_values,
                text_indicators, text_sparse)
    
    def process_event_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process event-associated data into flat numpy arrays.
        
        Args:
            data: DataFrame with TimedeltaIndex containing event features
            
        Returns:
            Tuple of:
            - times: (n_timesteps,) float32
            - indicators: (n_timesteps, n_event_feats) float32
        """
        if data.empty:
            n_ts = 0
        else:
            n_ts = len(data)
        
        n_feats = self.dims.n_event_feats
        times = np.zeros(n_ts, dtype=np.float32)
        indicators = np.zeros((n_ts, n_feats), dtype=np.float32)
        
        if data.empty:
            return times, indicators
        
        data = data.copy()
        data.columns = [col.replace(' ', '_').replace('-', '_') for col in data.columns]
        
        for t, record in enumerate(data.itertuples(index=True, name='Record')):
            times[t] = record.Index / np.timedelta64(1, 'h')
            
            for f, feat in enumerate(self.event_feats):
                feat_name = feat.replace(' ', '_').replace('-', '_')
                cols = self._get_feature_columns(feat_name, record)
                if cols:
                    values = [getattr(record, c) for c in cols]
                    if not all(pd.isna(v) or v == '' for v in values):
                        indicators[t, f] = 1.0
        
        return times, indicators
    
    def process_static_data(
        self,
        data: Union[pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        """
        Process static data into a flat numpy array.
        
        Args:
            data: Series or DataFrame containing static features
            
        Returns:
            Concatenated array of shape (static_total_dim,) as float32
        """
        static_array = np.zeros(self.dims.static_total_dim, dtype=np.float32)
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            return static_array
        
        # Convert to namedtuple
        if isinstance(data, pd.Series):
            Record = namedtuple('StaticData', data.index)
            record = Record(*data)
        elif isinstance(data, pd.DataFrame):
            Record = namedtuple('StaticData', data.columns)
            record = Record(*data.iloc[0])
        else:
            return static_array
        
        offset = 0
        for f, feat in enumerate(self.static_feats):
            feat_name = feat.replace(' ', '_').replace('-', '_')
            feat_dim = self.dims.static_feat_dims[f]
            feat_type = self.var_properties[feat]['type']
            
            if hasattr(record, feat_name):
                value = getattr(record, feat_name)
                
                if feat_type == 'numeric':
                    if pd.notna(value):
                        static_array[offset] = float(value)
                
                elif feat_type == 'categorical':
                    if pd.notna(value):
                        cat_map = self.var_properties[feat].get('category_map', {})
                        if isinstance(value, str):
                            idx_map = {v: int(k) for k, v in cat_map.items()}
                            cat_idx = idx_map.get(value, 0)
                        else:
                            cat_idx = int(value)
                        first_idx = min(int(k) for k in cat_map.keys()) if cat_map else 0
                        cat_idx = cat_idx - first_idx + 1
                        static_array[offset] = float(cat_idx)
                
                elif feat_type == 'text':
                    if pd.notna(value) and value.strip():
                        tokenized = self.tokenizer.process_text(value)
                        # Store token IDs as floats (will be cast later if needed)
                        static_array[offset:offset + feat_dim] = tokenized['input_ids'].astype(np.float32)
            
            offset += feat_dim
        
        return static_array
    
    def _get_feature_columns(self, base_name: str, record: namedtuple) -> List[str]:
        """Find columns matching a feature name in a namedtuple."""
        fields = record._fields if hasattr(record, '_fields') else []
        return [f for f in fields if re.match(f'^{re.escape(base_name)}(_\\d+)?$', f)]


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
    text_feats: Optional[List[str]],
    static_feats: List[str],
    dims_dict: dict
):
    """
    Initialize worker process with TensorizedDataProcessor.
    
    Called once per worker when the process pool is created.
    """
    global _tensorized_processor, _tensorized_dims
    
    # Reconstruct TensorDimensions from dict (can't pickle dataclass directly in some cases)
    _tensorized_dims = TensorDimensions(**dims_dict)
    
    tokenizer = LLMTextProcessor() if text_feats else None
    
    _tensorized_processor = DataProcessor(
        var_properties_path=var_properties_path,
        valued_feats=valued_feats,
        event_feats=event_feats,
        text_feats=text_feats,
        static_feats=static_feats,
        dims=_tensorized_dims,
        tokenizer=tokenizer
    )


def _process_single_episode(
    i: int,
    reader: MIMICDataReader,
    max_history_len_steps: int,
    max_episode_len_steps: int,
    max_episode_len_hours: Optional[int],
    min_episode_len_steps: Optional[int],
    min_episode_len_hours: Optional[int]
) -> Optional[EpisodeData]:
    """
    Process a single episode into EpisodeData for tensor insertion.
    
    This function runs in a worker process and returns minimal numpy arrays
    that the main process will insert into pre-allocated tensors.
    
    Args:
        i: Index in the reader
        reader: MIMICDataReader instance
        max_history_len_steps: Maximum historic timesteps
        max_episode_len_steps: Maximum episode timesteps
        max_episode_len_hours: Maximum hours to include
        min_episode_len_steps: Minimum required timesteps
        min_episode_len_hours: Minimum required hours
        
    Returns:
        EpisodeData if episode passes filters, None otherwise
    """
    global _tensorized_processor, _tensorized_dims
    
    try:
        _, statics, val_data, event_data, text_data, targets = reader[i]
        targets = dict(zip(
            ['mortality', 'length_of_stay', 'phenotype'],
            [np.array(t) for t in targets]
        ))
        
        # Apply filtering criteria
        if min_episode_len_hours is not None:
            if targets['length_of_stay'] < min_episode_len_hours:
                return None
        
        # Check minimum timesteps
        if min_episode_len_steps is not None:
            min_ts = np.timedelta64(0, 'h')
            max_ts = np.timedelta64(max_episode_len_hours, 'h') if max_episode_len_hours else None
            
            if text_data is not None:
                merged = val_data.merge(text_data, how='outer', left_index=True, right_index=True)
                check_data = merged
            else:
                check_data = val_data
            
            if max_ts is not None:
                is_current = (min_ts <= check_data.index) & (check_data.index < max_ts)
            else:
                is_current = min_ts <= check_data.index
            
            if is_current.sum() < min_episode_len_steps:
                return None
        
        # Resample value-associated data to hourly
        val_data = val_data.set_index(val_data.index.ceil('h')).resample(
            '1h', closed='right', label='right'
        ).mean()
        val_data = val_data.dropna(axis=0, how='all')
        
        # Filter timeseries
        (val_data, event_data, text_data,
         val_history_len, event_history_len,
         _max_history_len) = filter_timeseries_records(
            val_data, event_data, text_data,
            max_history_len_steps, max_episode_len_steps,
            max_episode_len_hours
        )
        
        # Merge text with value data
        if text_data is not None:
            val_data = val_data.merge(text_data, how='outer', left_index=True, right_index=True)
            val_data.columns = [
                col.rsplit('_', 1)[0] if col.endswith(('_left', '_right')) else col
                for col in val_data.columns
            ]
        
        # Process into numpy arrays
        (val_times, num_ind, num_vals, cat_ind, cat_vals,
         txt_ind, txt_sparse) = (
            _tensorized_processor.process_valued_data(val_data)
        )

        event_times, event_ind = (
            _tensorized_processor.process_event_data(event_data)
        )

        static_arr = _tensorized_processor.process_static_data(statics)

        # Normalize length of stay
        if max_episode_len_hours is not None:
            los = targets['length_of_stay'] - max_episode_len_hours
        else:
            max_val_t = (
                val_times.max() if len(val_times) > 0 else 0
            )
            max_event_t = (
                event_times.max() if len(event_times) > 0 else 0
            )
            los = targets['length_of_stay'] - max(
                max_val_t, max_event_t
            )

        return EpisodeData(
            idx=i,
            val_len=len(val_times),
            val_history_len=val_history_len,
            event_len=len(event_times),
            event_history_len=event_history_len,
            val_times=val_times,
            val_numeric_indicators=num_ind,
            val_numeric_values=num_vals,
            val_categorical_indicators=cat_ind,
            val_categorical_values=cat_vals,
            val_text_indicators=txt_ind,
            val_text_sparse=txt_sparse,
            event_times=event_times,
            event_indicators=event_ind,
            static_data=static_arr,
            mortality=float(targets['mortality']),
            length_of_stay=float(los),
            phenotype=targets['phenotype'].astype(np.float32)
        )
        
    except Exception as e:
        print(f"Error processing episode {i}: {e}")
        return None


def _get_tensor_dimensions(
    var_properties_path: str,
    valued_feats: List[str],
    event_feats: List[str],
    text_feats: Optional[List[str]],
    static_feats: List[str],
    max_ts_len: int,
    n_episodes: int,
    phenotype_dim: int,
    max_token_length: int = MAX_TOKEN_LENGTH
) -> TensorDimensions:
    """
    Compute tensor dimensions from configuration for pre-allocation.
    
    This function reads variable properties and computes the exact dimensions
    needed for all output tensors, enabling single-allocation of the final
    tensorized dataset.
    
    Args:
        var_properties_path: Path to variable_properties.yaml
        valued_feats: List of value-associated feature names (numeric + categorical in valued_feats)
        event_feats: List of event-associated feature names
        text_feats: List of text feature names (may be None)
        static_feats: List of static feature names
        max_ts_len: Maximum timeseries length (history + episode)
        n_episodes: Total number of episodes to process
        phenotype_dim: Number of phenotype labels
        max_token_length: Maximum token sequence length for text
        
    Returns:
        TensorDimensions dataclass with all dimension information
    """
    with open(var_properties_path, 'r') as f:
        var_properties = yaml.safe_load(f)
    
    # Separate valued_feats by type
    numeric_feats = []
    categorical_feats = []
    for feat in valued_feats:
        feat_type = var_properties[feat]['type']
        if feat_type == 'numeric':
            numeric_feats.append(feat)
        elif feat_type == 'categorical':
            categorical_feats.append(feat)
    
    # Get dimensions for each feature type
    numeric_feat_dims = [var_properties[f]['size'] for f in numeric_feats]
    categorical_feat_dims = [var_properties[f]['size'] for f in categorical_feats]
    
    # Text features use max_token_length for their dimension
    text_feats = text_feats or []
    text_feat_dims = [max_token_length for _ in text_feats]
    
    # Static features - handle different types
    static_feat_dims = []
    for feat in static_feats:
        feat_type = var_properties[feat]['type']
        if feat_type == 'text':
            static_feat_dims.append(max_token_length)
        else:
            static_feat_dims.append(var_properties[feat]['size'])
    
    return TensorDimensions(
        n_episodes=n_episodes,
        max_ts_len_val=max_ts_len,
        max_ts_len_event=max_ts_len,
        n_numeric_feats=len(numeric_feats),
        n_categorical_feats=len(categorical_feats),
        n_text_feats=len(text_feats),
        n_event_feats=len(event_feats),
        numeric_feat_dims=numeric_feat_dims,
        categorical_feat_dims=categorical_feat_dims,
        text_feat_dims=text_feat_dims,
        static_feat_dims=static_feat_dims,
        static_total_dim=sum(static_feat_dims),
        phenotype_dim=phenotype_dim,
    )


def _get_phenotype_dim(phenotypes_listfile: str) -> int:
    """
    Get the number of phenotype labels from the listfile header.
    
    Args:
        phenotypes_listfile: Path to phenotyping_<partition>_listfile.csv
        
    Returns:
        Number of phenotype columns (excludes 'stay' and 'period_length')
    """
    with open(phenotypes_listfile, 'r') as f:
        header = f.readline().strip()
    columns = header.split(',')
    # Header: "stay,period_length,<phenotype1>,<phenotype2>,..."
    return len(columns) - 2


def _allocate_output_arrays(dims: TensorDimensions) -> Dict[str, np.ndarray]:
    """
    Pre-allocate output arrays as numpy (not torch).
    """
    arrays = {}
    
    n = dims.n_episodes
    ts_val = dims.max_ts_len_val
    ts_event = dims.max_ts_len_event
    
    # Value-associated data
    arrays['val_times'] = np.zeros((n, ts_val), dtype=np.float32)
    arrays['val_masks'] = np.zeros((n, ts_val), dtype=np.float32)
    
    arrays['val_numeric_indicators'] = np.zeros((n, ts_val, dims.n_numeric_feats), dtype=np.float32)
    arrays['val_numeric_values'] = [
        np.zeros((n, ts_val, dim), dtype=np.float32)
        for dim in dims.numeric_feat_dims
    ]
    
    arrays['val_categorical_indicators'] = np.zeros((n, ts_val, dims.n_categorical_feats), dtype=np.float32)
    arrays['val_categorical_values'] = [
        np.zeros((n, ts_val, dim), dtype=np.int64)
        for dim in dims.categorical_feat_dims
    ]
    
    arrays['val_text_indicators'] = np.zeros((n, ts_val, dims.n_text_feats), dtype=np.float32)
    
    # Sparse text - collect as lists, finalize later
    arrays['_text_values_lists'] = [[] for _ in range(dims.n_text_feats)]
    arrays['_text_masks_lists'] = [[] for _ in range(dims.n_text_feats)]
    arrays['_text_timesteps_lists'] = [[] for _ in range(dims.n_text_feats)]
    arrays['_text_counts'] = [[] for _ in range(dims.n_text_feats)]
    
    # Event data
    arrays['event_times'] = np.zeros((n, ts_event), dtype=np.float32)
    arrays['event_masks'] = np.zeros((n, ts_event), dtype=np.float32)
    arrays['event_indicators'] = np.zeros((n, ts_event, dims.n_event_feats), dtype=np.float32)
    
    # Static and targets
    arrays['static_data'] = np.zeros((n, dims.static_total_dim), dtype=np.float32)
    arrays['mortality'] = np.zeros(n, dtype=np.float32)
    arrays['length_of_stay'] = np.zeros(n, dtype=np.float32)
    arrays['phenotype'] = np.zeros((n, dims.phenotype_dim), dtype=np.float32)
    
    return arrays


def _finalize_sparse_text(
    arrays: Dict[str, np.ndarray],
    dims: TensorDimensions
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Convert sparse text lists to numpy arrays.
    """
    val_text_offsets = []
    val_text_values = []
    val_text_masks = []
    val_text_timesteps = []
    
    for f in range(dims.n_text_feats):
        counts = arrays['_text_counts'][f]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)
        val_text_offsets.append(offsets)
        
        if arrays['_text_values_lists'][f]:
            val_text_values.append(np.stack(arrays['_text_values_lists'][f], axis=0))
            val_text_masks.append(np.stack(arrays['_text_masks_lists'][f], axis=0))
            val_text_timesteps.append(np.array(arrays['_text_timesteps_lists'][f], dtype=np.int32))
        else:
            token_len = dims.text_feat_dims[f]
            val_text_values.append(np.zeros((0, token_len), dtype=np.int64))
            val_text_masks.append(np.zeros((0, token_len), dtype=np.float32))
            val_text_timesteps.append(np.zeros(0, dtype=np.int32))
    
    del arrays['_text_values_lists']
    del arrays['_text_masks_lists']
    del arrays['_text_timesteps_lists']
    del arrays['_text_counts']
    
    return val_text_offsets, val_text_values, val_text_masks, val_text_timesteps


def filter_timeseries_records(
        numeric_data: pd.DataFrame,
        event_data: pd.DataFrame,
        text_data: Optional[pd.DataFrame] = None,
        max_history_len: int = 0,
        max_episode_len: int = 100,
        max_episode_len_hours: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, int]:
    """Filter timeseries records by history and episode length constraints.

    Returns:
        Tuple of (numeric_data, event_data, text_data,
                  val_history_len, event_history_len, max_history_len)
        where val_history_len and event_history_len are the actual
        number of pre-admission timesteps retained for value-associated
        and event-associated data respectively, and max_history_len is
        the configured maximum history length (for computing left-pad
        offsets during array insertion).
    """

    def filter(df):

        df = df.sort_index()

        # Exclude records collected more than x hours into the
        # current ICU stay episode
        if max_episode_len_hours is not None:
            selected_records = (
                df.index < np.timedelta64(max_episode_len_hours, 'h')
            )
            df = df.loc[selected_records, :]

        # Get indices of up to x records from the current ICU stay
        # episode, starting from the earliest record
        episode_record_indices = np.where(
            df.index >= np.timedelta64(0, 'h')
        )[0]
        episode_len = min(
            len(episode_record_indices), max_episode_len
        )
        episode_record_indices = episode_record_indices[:episode_len]

        # Get indices of up to x most recent records that were
        # collected before the current ICU stay episode
        historic_record_indices = np.where(
            df.index < np.timedelta64(0, 'h')
        )[0]
        history_len = min(
            len(historic_record_indices), max_history_len
        )
        # Guard against -0 slice: array[-0:] returns the full array
        # instead of an empty slice, so explicitly handle zero case
        if history_len > 0:
            historic_record_indices = (
                historic_record_indices[-history_len:]
            )
        else:
            historic_record_indices = np.array([], dtype=int)

        # Combine to give the indices of records to keep
        keep = np.concatenate(
            (historic_record_indices, episode_record_indices)
        )

        return df.iloc[keep, :], history_len

    event_data, event_history_len = filter(event_data)

    # Text data will eventually be embedded and merged with numeric
    # feature data, so they must be filtered together to ensure that
    # the length of the merged timeseries does not exceed the maximum
    # allowed timeseries length.
    if text_data is not None:
        # Merge text and numeric data on timestamps; used for
        # filtering the numeric and text data
        merged = numeric_data.merge(
            text_data, how='outer',
            left_index=True, right_index=True, indicator=True
        )
        # Get the names of the numeric and text features so that
        # they can be recovered after filtering
        numeric_feats = numeric_data.columns
        text_feats = text_data.columns
        filtered, val_history_len = filter(merged)
        # Split the filtered numeric and text data back into
        # separate DataFrames
        numeric_data = filtered.loc[
            filtered['_merge'].isin(['both', 'left_only']),
            numeric_feats
        ]
        text_data = filtered.loc[
            filtered['_merge'].isin(['both', 'right_only']),
            text_feats
        ]
    else:
        # No text data, so there's no need to merge timesteps
        numeric_data, val_history_len = filter(numeric_data)

    return (numeric_data, event_data, text_data,
            val_history_len, event_history_len, max_history_len)


def collate_tensorized(
    batch: MixedDataset,
    use_historical_records: bool = True,
    max_history_len_steps: int = 0
) -> MixedTensorDataset:
    """Collate function for MixedDataset.

    Takes a list of episode dicts and stacks them into the
    MixedTensorDataset format expected by the model. Pre-computed text
    embeddings are stacked into a single tensor at
    batch['val_data']['text']['embedded_values'] with shape
    [batch_size, max_ts_len, n_text_feats, embed_dim].

    Args:
        batch: List of episode dicts from MixedDataset.__getitem__.
        use_historical_records: If False, zero out masks for the
            history region [0, max_history_len_steps) so the model
            ignores pre-admission data. Defaults to True.
        max_history_len_steps: Number of timestep indices reserved
            for historical records. Only used when
            use_historical_records is False.
    """

    # Stack simple tensors directly
    val_times = torch.stack([b['val_times'] for b in batch], dim=0)
    val_masks = torch.stack([b['val_masks'] for b in batch], dim=0)
    event_times = torch.stack([b['event_times'] for b in batch], dim=0)
    event_masks = torch.stack([b['event_masks'] for b in batch], dim=0)

    # Mask out history region when historical records are disabled
    if not use_historical_records and max_history_len_steps > 0:
        val_masks[:, :max_history_len_steps] = 0.0
        event_masks[:, :max_history_len_steps] = 0.0
    static_data = torch.stack([b['static_data'] for b in batch], dim=0)

    # Stack indicator tensors
    val_numeric_ind = torch.stack([b['val_numeric_indicators'] for b in batch], dim=0)
    val_categorical_ind = torch.stack([b['val_categorical_indicators'] for b in batch], dim=0)
    val_text_ind = torch.stack([b['val_text_indicators'] for b in batch], dim=0)
    event_ind = torch.stack([b['event_indicators'] for b in batch], dim=0)

    # Stack per-feature value tensors
    n_numeric_feats = len(batch[0]['val_numeric_values'])
    n_categorical_feats = len(batch[0]['val_categorical_values'])
    n_text_feats = len(batch[0]['val_text_embeddings'])

    val_numeric_values = [
        torch.stack([b['val_numeric_values'][f] for b in batch], dim=0)
        for f in range(n_numeric_feats)
    ]
    val_categorical_values = [
        torch.stack([b['val_categorical_values'][f] for b in batch], dim=0)
        for f in range(n_categorical_feats)
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
    mortality = torch.stack([b['mortality'] for b in batch], dim=0).unsqueeze(-1)
    length_of_stay = torch.stack([b['length_of_stay'] for b in batch], dim=0).unsqueeze(-1)
    phenotype = torch.stack([b['phenotype'] for b in batch], dim=0)

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
            'mortality': mortality,
            'length_of_stay': length_of_stay,
            'phenotype': phenotype,
        },
    }

    if val_text_embeddings is not None:
        result['val_data']['text'] = {
            'indicators': val_text_ind,
            'embedded_values': val_text_embeddings,
        }

    return result


def save_dataset(dataset: MixedDataset, base_path: str) -> None:
    """
    Save tensorized dataset as directory of .npy files (memory-mappable).
    """
    os.makedirs(base_path, exist_ok=True)

    def save_array(name: str, arr: np.ndarray):
        np.save(os.path.join(base_path, f'{name}.npy'), arr)

    # Dense arrays
    save_array('val_numeric_indicators', dataset.val_numeric_indicators)
    save_array('val_categorical_indicators', dataset.val_categorical_indicators)
    save_array('val_text_indicators', dataset.val_text_indicators)
    save_array('val_times', dataset.val_times)
    save_array('val_masks', dataset.val_masks)
    save_array('event_indicators', dataset.event_indicators)
    save_array('event_times', dataset.event_times)
    save_array('event_masks', dataset.event_masks)
    save_array('static_data', dataset.static_data)
    save_array('mortality', dataset.mortality)
    save_array('length_of_stay', dataset.length_of_stay)
    save_array('phenotype', dataset.phenotype)

    # Per-feature arrays
    for i, arr in enumerate(dataset.val_numeric_values):
        save_array(f'val_numeric_values_{i}', arr)
    for i, arr in enumerate(dataset.val_categorical_values):
        save_array(f'val_categorical_values_{i}', arr)
    for i, arr in enumerate(dataset.val_text_offsets):
        save_array(f'val_text_offsets_{i}', arr)
    for i, arr in enumerate(dataset.val_text_values):
        save_array(f'val_text_values_{i}', arr)
    for i, arr in enumerate(dataset.val_text_masks):
        save_array(f'val_text_masks_{i}', arr)
    for i, arr in enumerate(dataset.val_text_timesteps):
        save_array(f'val_text_timesteps_{i}', arr)
    for i, arr in enumerate(dataset.val_text_embeddings):
        save_array(f'val_text_embeddings_{i}', arr)

    # Metadata
    metadata = {
        'max_ts_len': dataset.max_ts_len,
        'text_token_len': dataset.text_token_len,
        'text_embed_dim': dataset.text_embed_dim,
        'n_numeric_feats': len(dataset.val_numeric_values),
        'n_categorical_feats': len(dataset.val_categorical_values),
        'n_text_feats': dataset.n_text_feats,
    }
    with open(os.path.join(base_path, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Saved tensorized dataset to {base_path}/")


def load_dataset(base_path: str) -> MixedDataset:
    """
    Load tensorized dataset with memory-mapped arrays.

    Backward-compatible: if pre-computed embedding files are not present
    (old datasets created before the pre-embedding overhaul), empty lists
    are used for val_text_embeddings and text_embed_dim defaults to 0.
    The training script will fail fast if it expects embeddings and they
    are missing.
    """
    def load_mmap(name: str) -> np.ndarray:
        return np.load(os.path.join(base_path, f'{name}.npy'), mmap_mode='r')

    with open(os.path.join(base_path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    n_num = metadata['n_numeric_feats']
    n_cat = metadata['n_categorical_feats']
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
    )


def standardize_feats(
    arrays: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    dims: TensorDimensions,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None
) -> None:
    """Scale and center numeric feature values using their mean and the 5th-95th percentile range.

    This function standardizes the observed values of numeric features in-place. If `load_path` 
    is provided, the function loads the 5th and 95th percentiles and means from a .npz file and 
    uses them for standardization. If `load_path` is not provided, the function calculates the 
    means and percentiles from observed values (where indicator == 1.0) across all episodes and 
    timesteps. If `save_path` is provided, the calculated percentiles and means are saved to a 
    .npz file.

    Args:
        arrays (Dict[str, Union[np.ndarray, List[np.ndarray]]]): Dictionary containing pre-allocated numpy arrays with 
            keys 'val_numeric_indicators' (observation masks) and 'val_numeric_values' (list of per-feature value arrays). The value arrays are modified in-place.
        dims (TensorDimensions): Dataclass containing tensor dimension information, specifically 
            `n_numeric_feats` for the number of features to process.
        save_path (str, optional): Path to save the calculated percentiles and means to a .npz 
            file.
        load_path (str, optional): Path to load pre-calculated percentiles and means from a .npz 
            file.

    Returns:
        None. The arrays['val_numeric_values'] list is modified in-place with standardized values.
    """

    n_feats = dims.n_numeric_feats
    
    if load_path is not None:
        data = np.load(load_path)
        means = data['means']
        p5 = data['p5']
        p95 = data['p95']
    else:
        means = np.zeros(n_feats, dtype=np.float32)
        p5 = np.zeros(n_feats, dtype=np.float32)
        p95 = np.zeros(n_feats, dtype=np.float32)
        
        indicators = arrays['val_numeric_indicators']
        
        for f in range(n_feats):
            values = arrays['val_numeric_values'][f]
            mask = indicators[:, :, f] == 1.0
            
            if mask.any():
                observed = values[mask]
                means[f] = observed.mean()
                norms = np.linalg.norm(observed, ord=2, axis=-1)
                p5[f] = np.percentile(norms, 5)
                p95[f] = np.percentile(norms, 95)
        
        if save_path is not None:
            np.savez(save_path, means=means, p5=p5, p95=p95)
    
    for f in range(n_feats):
        if p5[f] == p95[f]:
            arrays['val_numeric_values'][f][:] = 0
        else:
            arrays['val_numeric_values'][f] -= means[f]
            arrays['val_numeric_values'][f] /= (p95[f] - p5[f])


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


def extract_mimic(
    reader: MIMICDataReader,
    suffix: str,
    output_dir: str,
    var_properties_path: str,
    max_episode_len_steps: int,
    max_history_len_steps: int = 0,
    min_episode_len_steps: Optional[int] = 10,
    min_episode_len_hours: Optional[int] = 48,
    max_episode_len_hours: Optional[int] = 48,
    n_workers: Optional[int] = None
) -> None:
    """
    Extract MIMIC ICU stay timeseries data from CSV files into a tensorized format optimized for 
    fast data loading during model training.

    This function reads patient episode data using a MIMICDataReader, applies filtering criteria,
    processes the data into pre-allocated numpy arrays, standardizes numeric features, and saves 
    the result as a directory of memory-mappable .npy files. The output format is designed for 
    efficient multi-worker DataLoader access with minimal memory overhead.

    The extraction pipeline operates in two passes:
        1. **Filtering pass**: Episodes are processed in parallel to apply inclusion criteria 
           (minimum length, minimum timesteps) and extract raw data.
        2. **Insertion pass**: Surviving episodes are inserted into pre-allocated arrays and 
           text features are converted to sparse storage format.

    Output Structure:
        The function creates a directory at `{output_dir}/{suffix}/` containing:
        
        * Dense arrays (one .npy file each):
            - `val_times.npy`: (n_episodes, max_ts_len) float32 - Timestamps in hours
            - `val_masks.npy`: (n_episodes, max_ts_len) float32 - Valid timestep indicators
            - `val_numeric_indicators.npy`: (n_episodes, max_ts_len, n_numeric_feats) float32
            - `val_categorical_indicators.npy`: (n_episodes, max_ts_len, n_cat_feats) float32
            - `val_text_indicators.npy`: (n_episodes, max_ts_len, n_text_feats) float32
            - `event_times.npy`: (n_episodes, max_ts_len) float32
            - `event_masks.npy`: (n_episodes, max_ts_len) float32
            - `event_indicators.npy`: (n_episodes, max_ts_len, n_event_feats) float32
            - `static_data.npy`: (n_episodes, static_total_dim) float32
            - `mortality.npy`: (n_episodes,) float32
            - `length_of_stay.npy`: (n_episodes,) float32
            - `phenotype.npy`: (n_episodes, n_phenotypes) float32
        
        * Per-feature arrays:
            - `val_numeric_values_{i}.npy`: (n_episodes, max_ts_len, feat_dim) float32
            - `val_categorical_values_{i}.npy`: (n_episodes, max_ts_len, feat_dim) int64
        
        * Sparse text arrays (CSR-style storage for memory efficiency):
            - `val_text_offsets_{i}.npy`: (n_episodes + 1,) int64 - CSR row pointers
            - `val_text_values_{i}.npy`: (n_non_empty, token_len) int64 - Token IDs
            - `val_text_masks_{i}.npy`: (n_non_empty, token_len) float32 - Attention masks
            - `val_text_timesteps_{i}.npy`: (n_non_empty,) int32 - Original timestep indices
        
        * Metadata:
            - `metadata.pkl`: Dictionary containing dimension information for reconstruction
        
        Additionally creates:
            - `{suffix}_ids.pkl`: List of patient episode IDs that passed filtering
            - `summary_statistics_train.npz`: (train only) Standardization parameters.

    Standardization:
        Numeric features are standardized using mean centering and percentile-based scaling.
        For the training set, statistics (mean, 5th percentile, 95th percentile) are computed
        and saved to `summary_statistics_train.npz`. Validation and test sets are standardized 
        using the training set statistics, which must be extracted first.

    Args:
        reader (MIMICDataReader): Configured data reader for the target partition. Must have 
            `prediction_task='all'` to include all target labels.
        suffix (str): Data partition identifier ('train', 'val', or 'test'). Determines output 
            filenames and whether to compute or load standardization statistics.
        output_dir (str): Directory where the tensorized dataset will be saved. Created if it 
            does not exist.
        var_properties_path (str): Path to variable_properties.yaml containing feature type 
            information and category mappings.
        max_episode_len_steps (int): Maximum number of timesteps to include from each ICU 
            episode, counted from admission time.
        max_history_len_steps (int, optional): Maximum number of pre-admission timesteps to 
            include, counted backwards from admission. Defaults to 0.
        min_episode_len_steps (int, optional): Minimum required timesteps within the inclusion 
            window for an episode to be included. Episodes with fewer timesteps are filtered 
            out. Defaults to 10. Set to None to disable this filter.
        min_episode_len_hours (int, optional): Minimum ICU length of stay in hours for episode 
            inclusion. Defaults to 48. Set to None to disable this filter.
        max_episode_len_hours (int, optional): Maximum hours from admission to include in 
            extracted data. Records after this time are excluded. Defaults to 48. Set to None 
            to include all available records.
        n_workers (int, optional): Number of parallel worker processes for data extraction. 
            Each worker holds a tokenizer instance for text processing. Defaults to 1.

    Raises:
        ValueError: If reader.prediction_task is not 'all'.
        FileNotFoundError: If extracting validation or test data before training data 
            (standardization statistics are required).

    Notes:
        - Text features use sparse storage because most timesteps lack text data. The CSR-style
          format stores only non-empty entries, reducing storage from O(n_episodes * max_ts_len * 
          token_len) to O(n_non_empty * token_len).
        - The output arrays are saved as separate .npy files to enable memory-mapped loading,
          which allows multi-worker DataLoaders to share read-only memory efficiently.
        - Value-associated data is resampled to hourly resolution before extraction.
        - Length of stay targets are normalized relative to max_episode_len_hours (if set) or
          the maximum observed timestamp.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if reader.prediction_task != 'all':
        raise ValueError(f"reader.prediction_task must be 'all', got {reader.prediction_task}")
    
    total_episodes = len(reader.patient_episode_ids)
    max_ts_len = max_history_len_steps + max_episode_len_steps
    
    if n_workers is None:
        n_workers = 1
    
    print(f"Processing {total_episodes} episodes using {n_workers} workers...")
    sys.stdout.flush()
    
    # Get phenotype dimension
    phenotype_dim = _get_phenotype_dim(reader.phenotypes_listfile)
    
    # Compute tensor dimensions
    dims = _get_tensor_dimensions(
        var_properties_path=var_properties_path,
        valued_feats=reader.valued_feats,
        event_feats=reader.event_feats,
        text_feats=reader.text_feats,
        static_feats=reader.static_feats,
        max_ts_len=max_ts_len,
        n_episodes=total_episodes,  # Will be updated after filtering
        phenotype_dim=phenotype_dim
    )
    
    # Convert dims to dict for pickling to workers
    dims_dict = {
        'n_episodes': dims.n_episodes,
        'max_ts_len_val': dims.max_ts_len_val,
        'max_ts_len_event': dims.max_ts_len_event,
        'n_numeric_feats': dims.n_numeric_feats,
        'n_categorical_feats': dims.n_categorical_feats,
        'n_text_feats': dims.n_text_feats,
        'n_event_feats': dims.n_event_feats,
        'numeric_feat_dims': dims.numeric_feat_dims,
        'categorical_feat_dims': dims.categorical_feat_dims,
        'text_feat_dims': dims.text_feat_dims,
        'static_feat_dims': dims.static_feat_dims,
        'static_total_dim': dims.static_total_dim,
        'phenotype_dim': dims.phenotype_dim,
    }
    
    # Process episodes in parallel, collecting results
    process_fn = partial(
        _process_single_episode,
        reader=reader,
        max_history_len_steps=max_history_len_steps,
        max_episode_len_steps=max_episode_len_steps,
        max_episode_len_hours=max_episode_len_hours,
        min_episode_len_steps=min_episode_len_steps,
        min_episode_len_hours=min_episode_len_hours
    )
    
    # Collect results in a first pass to count surviving episodes
    results = []
    n_ignored = 0
    
    print("Pass 1: Processing episodes and filtering...")
    sys.stdout.flush()
    
    with mp.Pool(
        processes=n_workers,
        initializer=_init_tensorized_worker,
        initargs=(var_properties_path, reader.valued_feats, reader.event_feats,
                  reader.text_feats, reader.static_feats, dims_dict)
    ) as pool:
        for result in tqdm(
            pool.imap(process_fn, range(total_episodes), chunksize=10),
            total=total_episodes,
            desc=f"Extracting {suffix}"
        ):
            if result is None:
                n_ignored += 1
            else:
                results.append(result)
    
    n_valid = len(results)
    print(f"Extracted {n_valid} episodes, ignored {n_ignored} that didn't meet criteria.")
    sys.stdout.flush()
    
    # Update dims with actual episode count
    dims = TensorDimensions(
        n_episodes=n_valid,
        max_ts_len_val=dims.max_ts_len_val,
        max_ts_len_event=dims.max_ts_len_event,
        n_numeric_feats=dims.n_numeric_feats,
        n_categorical_feats=dims.n_categorical_feats,
        n_text_feats=dims.n_text_feats,
        n_event_feats=dims.n_event_feats,
        numeric_feat_dims=dims.numeric_feat_dims,
        categorical_feat_dims=dims.categorical_feat_dims,
        text_feat_dims=dims.text_feat_dims,
        static_feat_dims=dims.static_feat_dims,
        static_total_dim=dims.static_total_dim,
        phenotype_dim=dims.phenotype_dim,
    )
    
    # Allocate output arrays
    print("Allocating output arrays...")
    sys.stdout.flush()
    arrays = _allocate_output_arrays(dims)
    
    # Insert results into arrays
    print("Pass 2: Inserting data into arrays...")
    sys.stdout.flush()
    
    valid_ids = []
    for out_idx, ep in enumerate(tqdm(results, desc="Building arrays")):
        valid_ids.append(reader.patient_episode_ids[ep.idx])
        val_len = ep.val_len
        val_hist = ep.val_history_len
        val_ep = val_len - val_hist
        event_len = ep.event_len
        event_hist = ep.event_history_len
        event_ep = event_len - event_hist

        # Left-pad history: actual history data is right-justified
        # within [0, max_history_len_steps). Episode data starts at
        # index max_history_len_steps.
        vh_start = max_history_len_steps - val_hist
        ve_start = max_history_len_steps
        eh_start = max_history_len_steps - event_hist
        ee_start = max_history_len_steps

        if val_len > 0:
            # History portion (right-justified in history region)
            if val_hist > 0:
                arrays['val_times'][
                    out_idx, vh_start:max_history_len_steps
                ] = ep.val_times[:val_hist]
                arrays['val_masks'][
                    out_idx, vh_start:max_history_len_steps
                ] = 1.0
                arrays['val_numeric_indicators'][
                    out_idx, vh_start:max_history_len_steps, :
                ] = ep.val_numeric_indicators[:val_hist]
                for f, vals in enumerate(ep.val_numeric_values):
                    arrays['val_numeric_values'][f][
                        out_idx, vh_start:max_history_len_steps, :
                    ] = vals[:val_hist]
                arrays['val_categorical_indicators'][
                    out_idx, vh_start:max_history_len_steps, :
                ] = ep.val_categorical_indicators[:val_hist]
                for f, vals in enumerate(ep.val_categorical_values):
                    arrays['val_categorical_values'][f][
                        out_idx, vh_start:max_history_len_steps, :
                    ] = vals[:val_hist]
                arrays['val_text_indicators'][
                    out_idx, vh_start:max_history_len_steps, :
                ] = ep.val_text_indicators[:val_hist]

            # Episode portion (starts at max_history_len_steps)
            if val_ep > 0:
                arrays['val_times'][
                    out_idx, ve_start:ve_start + val_ep
                ] = ep.val_times[val_hist:]
                arrays['val_masks'][
                    out_idx, ve_start:ve_start + val_ep
                ] = 1.0
                arrays['val_numeric_indicators'][
                    out_idx, ve_start:ve_start + val_ep, :
                ] = ep.val_numeric_indicators[val_hist:]
                for f, vals in enumerate(ep.val_numeric_values):
                    arrays['val_numeric_values'][f][
                        out_idx, ve_start:ve_start + val_ep, :
                    ] = vals[val_hist:]
                arrays['val_categorical_indicators'][
                    out_idx, ve_start:ve_start + val_ep, :
                ] = ep.val_categorical_indicators[val_hist:]
                for f, vals in enumerate(ep.val_categorical_values):
                    arrays['val_categorical_values'][f][
                        out_idx, ve_start:ve_start + val_ep, :
                    ] = vals[val_hist:]
                arrays['val_text_indicators'][
                    out_idx, ve_start:ve_start + val_ep, :
                ] = ep.val_text_indicators[val_hist:]

            # Sparse text — iterate over non-empty entries and remap
            # timestep indices to left-padded layout
            for f in range(dims.n_text_feats):
                for (t, token_ids, mask) in ep.val_text_sparse[f]:
                    if t < val_hist:
                        mapped_t = vh_start + t
                    else:
                        mapped_t = ve_start + (t - val_hist)
                    arrays['_text_values_lists'][f].append(token_ids)
                    arrays['_text_masks_lists'][f].append(mask)
                    arrays['_text_timesteps_lists'][f].append(
                        mapped_t
                    )
                arrays['_text_counts'][f].append(
                    len(ep.val_text_sparse[f])
                )
        else:
            for f in range(dims.n_text_feats):
                arrays['_text_counts'][f].append(0)

        if event_len > 0:
            # History portion (right-justified in history region)
            if event_hist > 0:
                arrays['event_times'][
                    out_idx, eh_start:max_history_len_steps
                ] = ep.event_times[:event_hist]
                arrays['event_masks'][
                    out_idx, eh_start:max_history_len_steps
                ] = 1.0
                arrays['event_indicators'][
                    out_idx, eh_start:max_history_len_steps, :
                ] = ep.event_indicators[:event_hist]

            # Episode portion (starts at max_history_len_steps)
            if event_ep > 0:
                arrays['event_times'][
                    out_idx, ee_start:ee_start + event_ep
                ] = ep.event_times[event_hist:]
                arrays['event_masks'][
                    out_idx, ee_start:ee_start + event_ep
                ] = 1.0
                arrays['event_indicators'][
                    out_idx, ee_start:ee_start + event_ep, :
                ] = ep.event_indicators[event_hist:]

        arrays['static_data'][out_idx, :] = ep.static_data
        arrays['mortality'][out_idx] = ep.mortality
        arrays['length_of_stay'][out_idx] = ep.length_of_stay
        arrays['phenotype'][out_idx, :] = ep.phenotype
    
    # Finalize sparse text storage
    print("Finalizing sparse text storage...")
    sys.stdout.flush()
    (val_text_offsets, val_text_values, 
     val_text_masks, val_text_timesteps) = _finalize_sparse_text(arrays, dims)
    
    # Free results memory
    del results
    gc.collect()
    
    # Standardize numeric features
    summary_stats_path = os.path.join(output_dir, 'summary_statistics_train.npz')
    if suffix == 'train':
        print("Computing and applying standardization...")
        sys.stdout.flush()
        standardize_feats(arrays, dims, save_path=summary_stats_path)
    else:
        if not os.path.exists(summary_stats_path):
            raise FileNotFoundError(
                "summary_statistics_train.npz not found. Run training extraction first."
            )
        print("Loading and applying standardization...")
        sys.stdout.flush()
        standardize_feats(arrays, dims, load_path=summary_stats_path)
    
    # Create dataset and save
    print("Saving dataset...")
    sys.stdout.flush()
    
    # Create dataset with sparse text (embeddings will be added later by embed_text.py)
    dataset = MixedDataset(
        val_numeric_indicators=arrays['val_numeric_indicators'],
        val_numeric_values=arrays['val_numeric_values'],
        val_categorical_indicators=arrays['val_categorical_indicators'],
        val_categorical_values=arrays['val_categorical_values'],
        val_text_indicators=arrays['val_text_indicators'],
        val_times=arrays['val_times'],
        val_masks=arrays['val_masks'],
        # Sparse text (tokens for XAI, embeddings added by embed_text.py)
        val_text_offsets=val_text_offsets,
        val_text_values=val_text_values,
        val_text_masks=val_text_masks,
        val_text_timesteps=val_text_timesteps,
        val_text_embeddings=[],  # Populated by embed_text.py
        text_embed_dim=0,  # Set by embed_text.py
        # Event data
        event_indicators=arrays['event_indicators'],
        event_times=arrays['event_times'],
        event_masks=arrays['event_masks'],
        # Static and targets
        static_data=arrays['static_data'],
        mortality=arrays['mortality'],
        length_of_stay=arrays['length_of_stay'],
        phenotype=arrays['phenotype'],
        # Metadata
        max_ts_len=dims.max_ts_len_val,
        text_token_len=dims.text_feat_dims,
    )
    
    output_path = os.path.join(output_dir, f'{suffix}')
    save_dataset(dataset, output_path)
    
    # Also save IDs for reference
    ids_path = os.path.join(output_dir, f'{suffix}_ids.pkl')
    with open(ids_path, 'wb') as f:
        pickle.dump(valid_ids, f)
    
    print(f"Tensorized {suffix} data saved to {output_path}")
    print(f"Episode IDs saved to {ids_path}\n")


def prepare_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    balance_text: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    use_historical_records: bool = True,
    max_history_len_steps: int = 0
) -> List[DataLoader]:
    """Prepare training, (validation), and test DataLoaders for MixedDataset.

    This function creates PyTorch DataLoader instances for `MixedDataset` objects prepared by 
    `extract_mimic()`. The dataset uses memory-mapped numpy arrays for efficient multi-worker 
    access with minimal memory overhead. Workers share read-only memory-mapped arrays rather 
    than duplicating data in each worker process' memory space.

    The function loads pre-extracted datasets from `{data_dir}/{partition}/` directories, where 
    each directory contains:
        - Dense arrays for feature indicators, times, and masks
        - Per-feature .npy arrays for numeric and categorical feature values
        - Sparse arrays in CSR-style format for text features
        - metadata.pkl with dimension information
    
    Dataloaders are configured for efficient GPU training with configurable worker processes, 
    memory pinning, and batch prefetching. The training loader shuffles data while validation 
    and test loaders maintain sequential order.

    For distributed training with text features, the optional `balance_text` parameter enables
    text-balanced sampling across ranks for all partitions (train, val, test). This prevents 
    memory imbalance where one GPU receives all text-heavy episodes and OOMs while others have 
    light batches. Within each meta-batch (batch_size * world_size samples), episodes are sorted 
    by text density and distributed via round-robin to ensure each rank gets a mix of text-heavy 
    and text-light episodes. For training data, global shuffling is preserved - only 
    within-meta-batch distribution is deterministic. For validation and test data, ordering is 
    deterministic but balanced across ranks.

    The function supports three modes:
        1. Single-GPU: No sampler, standard shuffling for train
        2. Multi-GPU without balancing: Use accelerator.prepare_data_loader() after calling this
        3. Multi-GPU with balancing: Pass balance_text=True with world_size and rank

    In multi-GPU setups, a good rule of thumb is to have at least num_workers * num_gpus CPUs, 
    and double that if possible.

    Args:
        data_dir (str): Directory containing 'train/', 'val/', and 'test/' subdirectories. Each 
            subdirectory should be the output of `extract_mimic()`.
        batch_size (int): Number of samples per batch (per GPU in distributed settings).
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfers. 
            Defaults to True. Only effective if num_workers > 0.
        prefetch_factor (int, optional): Number of batches to prefetch per worker. Defaults to 2. 
            Higher values increase memory usage but can improve throughput if batch processing by 
            the model is slower than data loading. Only effective if num_workers > 0.
        balance_text (bool, optional): If True and running distributed (world_size > 1), use 
            TextBalancedDistributedSampler to balance text density across ranks for all partitions.
            This prevents memory imbalance in distributed training with sparse text data. 
            Defaults to False. When False, no distributed sampler is added - use 
            accelerator.prepare_data_loader() to add standard distributed sampling.
        world_size (int, optional): Number of distributed processes. Required if balance_text=True.
            Can be obtained from accelerator.num_processes.
        rank (int, optional): Current process rank. Required if balance_text=True. Can be obtained
            from accelerator.process_index.
        use_historical_records (bool, optional): If False, zero out masks for the history region
            [0, max_history_len_steps) so the model ignores pre-admission data. Defaults to True.
        max_history_len_steps (int, optional): Number of timestep indices reserved for historical
            records in the extracted arrays. Only used when use_historical_records is False.
            Defaults to 0.

    Returns:
        List[DataLoader]: List of DataLoaders in order: [train_loader, val_loader (if available), 
            test_loader]. If validation data are not found, only [train_loader, test_loader] is 
            returned.
    
    Raises:
        FileNotFoundError: If 'train/' or 'test/' directories are not found in `data_dir`.
        ValueError: If balance_text=True but world_size or rank is not provided.
    
    Note:
        When using balance_text=True:
        - Do NOT wrap the returned dataloaders with accelerator.prepare_data_loader() as the 
          custom sampler already handles distributed sampling.
        - Call train_loader.sampler.set_epoch(epoch) at the start of each training epoch to 
          ensure proper shuffling.
        
        When using balance_text=False for distributed training:
        - Wrap dataloaders with accelerator.prepare_data_loader() to add distributed sampling.
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
        
        collate_fn = partial(
            collate_tensorized,
            use_historical_records=use_historical_records,
            max_history_len_steps=max_history_len_steps,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            multiprocessing_context='spawn' if num_workers > 0 else None
        )
        
        dataloaders.append(loader)
    
    return dataloaders
    