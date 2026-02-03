import numpy as np

from dataclasses import dataclass
from numpy import ndarray
from torch import Tensor
from typing import Dict, List, NamedTuple, Union


EventAssociatedDataEntry = Dict[str, List[List[ndarray]]]
StaticDataEntry = List[ndarray]
ValueAssociatedDataEntry = Dict[str, Dict[str, List[List[ndarray]]]]
TargetDataEntry = Dict[str, ndarray]

# Data types created by preprocessing functions that act on MixedDataset. Used as input to models.
EventAssociatedTensorData = Dict[str, Tensor]
StaticTensorData = Tensor
ValueAssociatedTensorData = Dict[str, Union[Dict[str, Union[Tensor, List[Tensor]]], Tensor]]
TargetTensorData = Dict[str, Tensor]

MixedTensorDataset = Dict[
    str, Union[ValueAssociatedTensorData, EventAssociatedTensorData, StaticTensorData, TargetTensorData]
]

@dataclass
class TensorDimensions:
    """
    Stores pre-computed tensor dimensions for tensorized dataset allocation.
    
    These dimensions are derived from the dataset configuration and variable properties,
    allowing pre-allocation of output tensors before processing begins.
    
    Attributes:
        n_episodes: Total number of patient episodes in the dataset
        max_ts_len_val: Maximum timesteps for value-associated data
        max_ts_len_event: Maximum timesteps for event-associated data
        n_numeric_feats: Number of numeric features
        n_categorical_feats: Number of categorical features
        n_text_feats: Number of text features
        n_event_feats: Number of event features
        numeric_feat_dims: List of dimensions for each numeric feature
        categorical_feat_dims: List of dimensions for each categorical feature
        text_feat_dims: List of token sequence lengths for each text feature
        static_feat_dims: List of dimensions for each static feature
        static_total_dim: Total dimension of concatenated static features
        phenotype_dim: Number of phenotype labels
    """
    n_episodes: int
    max_ts_len_val: int
    max_ts_len_event: int
    n_numeric_feats: int
    n_categorical_feats: int
    n_text_feats: int
    n_event_feats: int
    numeric_feat_dims: list
    categorical_feat_dims: list
    text_feat_dims: list
    static_feat_dims: list
    static_total_dim: int
    phenotype_dim: int


class EpisodeData(NamedTuple):
    """
    Container for a single processed episode's data before tensor insertion.
    
    This is an intermediate format returned by worker processes during parallel
    extraction. Data is stored as numpy arrays with minimal padding, then inserted
    into pre-allocated tensors by the main process.
    
    Attributes:
        idx: Original index in the reader (for ID lookup)
        val_len: Actual number of value-associated timesteps (before padding)
        event_len: Actual number of event timesteps (before padding)
        val_times: Array of timestamps for value-associated data, shape (val_len,)
        val_numeric_indicators: Array of shape (val_len, n_numeric_feats)
        val_numeric_values: List of arrays, each shape (val_len, feat_dim)
        val_categorical_indicators: Array of shape (val_len, n_categorical_feats)
        val_categorical_values: List of arrays, each shape (val_len, feat_dim)
        val_text_indicators: Array of shape (val_len, n_text_feats)
        val_text_values: List of arrays, each shape (val_len, token_len)
        val_text_masks: List of arrays, each shape (val_len, token_len)
        event_times: Array of timestamps for event data, shape (event_len,)
        event_indicators: Array of shape (event_len, n_event_feats)
        static_data: Array of shape (static_total_dim,)
        mortality: Scalar
        length_of_stay: Scalar
        phenotype: Array of shape (phenotype_dim,)
    """
    idx: int
    val_len: int
    event_len: int
    val_times: 'np.ndarray'
    val_numeric_indicators: 'np.ndarray'
    val_numeric_values: list
    val_categorical_indicators: 'np.ndarray'
    val_categorical_values: list
    val_text_indicators: 'np.ndarray'
    val_text_values: list
    val_text_masks: list
    event_times: 'np.ndarray'
    event_indicators: 'np.ndarray'
    static_data: 'np.ndarray'
    mortality: float
    length_of_stay: float
    phenotype: 'np.ndarray'
