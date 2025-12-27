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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

from TransEHR2.constants import HF_API_TOKEN, LLM_NAME, MAX_TOKEN_LENGTH, TOKENIZER_PAD_TOKEN
from TransEHR2.data.custom_types import EventAssociatedDataEntry, StaticDataEntry, ValueAssociatedDataEntry
from TransEHR2.data.custom_types import MixedTensorDataset
from TransEHR2.data.datareaders import MIMICDataReader
from TransEHR2.data.datasets import MixedDataset, HDF5Dataset


_worker_processor = None  # Global variable to hold DataProcessor instance in each worker in parallel processing


class LlamaTextProcessor:

    def __init__(self, model_name: str = LLM_NAME, max_length: int = MAX_TOKEN_LENGTH):
        """
        Initialize the LLAMA text processor.
        
        Args:
            model_name (str): The LLAMA model name to use for tokenization
            max_length (int): Maximum sequence length for tokenized text
        """

        # Use local files to avoid making too many requests and hitting rate limits
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
        self.tokenizer.add_special_tokens({'pad_token': TOKENIZER_PAD_TOKEN})
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
    A callable base class for preprocessing electronic health record data read from disk.
    
    This class prepares data so that it can be served by DataLoaders later on. It handles 
    different data types and feature types according to a variable properties map.
    """
    
    def __init__(
        self,
        max_timeseries_length: int,
        var_properties_path: Optional[str] = None,
        tokenizer: Optional[LlamaTextProcessor] = None
    ):
        """
        Initialize the DataProcessor.
        
        Args:
            max_timeseries_length (int): Maximum length of the timeseries data. This is used for padding.
            var_properties_path (str, optional): Path to the variable properties YAML file. If None, will look in default locations.
            tokenizer (LlamaTextProcessor, optional): Optional tokenizer for text features. Must be provided if text features are present.
        """

        self.max_timeseries_length = max_timeseries_length

        # Load variable properties map
        if var_properties_path is None:
            # Try to find the properties file in standard locations
            root = os.path.dirname(os.path.abspath(__file__))
            potential_paths = (
                os.path.join(root, 'variable_properties.yaml'),
                os.path.join(root, '..', 'data', 'variable_properties.yaml'),
                os.path.join(root, '..', '..', 'data', 'variable_properties.yaml')
            )
            
            var_properties_path = next(
                (path for path in potential_paths if os.path.exists(path)), None
            )
            
            if var_properties_path is None:
                raise FileNotFoundError(
                    "Could not find variable_properties.yaml in standard locations. Please provide an explicit path."
                )
        
        with open(var_properties_path, 'r') as f:
            self.var_properties = yaml.safe_load(f)

        self.tokenizer = tokenizer

    def __call__(
        self,
        data_type: str,
        data: pd.DataFrame,
        candidate_feats: List[str]
    ) -> Union[EventAssociatedDataEntry, ValueAssociatedDataEntry, StaticDataEntry]:
        """
        Extract timeseries data according to data_type and feature types.
        
        Args:
            data_type (str): One of 'value', 'event', or 'static'.
            data (pd.DataFrame): DataFrame containing timeseries data
            candidate_feats (list): List of feature names to extract
            
        Returns:
            Union[EventAssociatedDataEntry, ValueAssociatedDataEntry, StaticDataEntry]: Processed data that can be used as an entry in a MixedDataset.
        """

        if data_type == 'static':
            # Prepare a StaticDataEntry for a MixedDataset
            return self.extract_static_data(data, candidate_feats)

        # If data_type is 'value' or 'event', ensure the DataFrame has a TimedeltaIndex
        if not isinstance(data.index, pd.TimedeltaIndex):
            raise ValueError('Value and event data must have a pd.TimedeltaIndex')
        
        if data_type == 'value':
            # Prepare a ValueAssociatedDataEntry for a MixedDataset
            return self.extract_value_timeseries(data, candidate_feats)
        elif data_type == 'event':
            # Prepare an EventAssociatedDataEntry for a MixedDataset
            return self.extract_event_timeseries(data, candidate_feats)
        else:
            # Raise an error for unsupported data types
            raise ValueError(f"Unsupported data_type: {data_type}. Use 'value', 'event', or 'static'.")
        
    def _get_feature_type(self, feature_name: str) -> str:
        """
        Get the type of a feature from the variable properties map.
        
        Args:
            feature_name (str): Name of the feature
            
        Returns:
            str: Feature type ('numeric', 'categorical', or 'text')
        """
        if feature_name not in self.var_properties:
            raise KeyError(f"Feature '{feature_name}' not found in variable properties.")
        
        return self.var_properties[feature_name]['type']
    
    def _get_feature_size(self, feature_name: str) -> int:
        """
        Get the size of a feature from the variable properties map.
        
        Args:
            feature_name (str): Name of the feature
            
        Returns:
            int: Feature size (number of components)
        """
        if feature_name not in self.var_properties:
            raise KeyError(f"Feature '{feature_name}' not found in variable properties.")
        
        return self.var_properties[feature_name]['size']
    
    def _get_feature_column_names(self, base_name: str, data: Union[pd.DataFrame, namedtuple]) -> List[str]:
        """Get feature names from a DataFrame based on the provided base feature name.
        
        This method finds columns in the DataFrame that start with the base feature name. It is assumed that vector-valued features have one column per vector dimension, and the column names of vector-valued features are the base feature name followed by an underscore and the dimension index (e.g., 'feature_0', 'feature_1', etc.). Scalar features' column names are simply the base feature name without any suffix.

        Args:
            base_name (str): The base feature names to search for in the DataFrame.
            data (Union[pd.DataFrame, namedtuple]): The DataFrame or namedtuple to search for feature columns.
        
        Returns:
            List[str]: A list of column names that correspond to the feature.
        """
        feature_columns = []
        
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame case
            matching_columns = [col for col in data.columns if re.search(f'^{re.escape(base_name)}(_\d+)?$', col)]
            if matching_columns:
                feature_columns.extend(matching_columns)
        else:
            # Handle namedtuple case
            fields = data._fields if hasattr(data, '_fields') else []
            matching_columns = [f for f in fields if re.search(f'^{re.escape(base_name)}(_\d+)?$', f)]
            if matching_columns:
                feature_columns.extend(matching_columns)
                
        return feature_columns
    
    def _get_category_map(self, feature_name: str) -> Dict[int, str]:
        """
        Get the category map for a categorical feature.
        
        Args:
            feature_name (str): Name of the categorical feature
            
        Returns:
            dict: Dictionary mapping integer codes to category labels
        """
        if feature_name not in self.var_properties:
            raise KeyError(f"Feature '{feature_name}' not found in variable properties.")
        
        if self.var_properties[feature_name]['type'] != 'categorical':
            warn(f"Feature '{feature_name}' is not categorical. No category map available.")
            return {}
        
        return self.var_properties[feature_name]['category_map']

    def _set_category_map(self, feature_name: str, category_map: Dict[int, str]) -> None:
        """
        Update the category map for a categorical feature in the variable properties dictionary.
        
        Args:
            feature_name (str): Name of the categorical feature
            category_map (Dict[int, str]): New dictionary mapping integer codes to category labels
            
        Raises:
            KeyError: If feature_name does not exist in the variable properties
            ValueError: If the feature is not of categorical type
        """
        if feature_name not in self.var_properties:
            raise KeyError(f"Feature '{feature_name}' not found in variable properties.")
        
        if self.var_properties[feature_name]['type'] != 'categorical':
            raise ValueError(f"Cannot set category map for feature '{feature_name}' because it is not categorical.")
        
        # Update the category map
        self.var_properties[feature_name]['category_map'] = category_map
    
    def _extract_numeric(
        self,
        data: namedtuple,
        candidate_feats: List[str]
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        
        feat_indicator_list = []  # A list of all features' indicators for this timestep
        feat_value_list = []  # A list of all features' values for this timestep
        for ft in candidate_feats:
            ft_size = self._get_feature_size(ft)
            # Convert the candidate feature to valid Python identifiers that match the new column names
            ft = ft.replace(' ', '_').replace('-', '_')
            cols = sorted(self._get_feature_column_names(ft, data))
            # If the feature is present, extract its values. Otherwise, store a zero-valued array.
            if cols:
                if ft_size != len(cols):
                    raise ValueError(
                        f"The number of columns for '{ft}' does not match its expected dimension ({ft_size})"
                    )
                value = [getattr(data, col) for col in cols]
                # If the feature value is missing in all dims the indicator bit is set to zero
                if all([pd.isna(v) for v in value]):
                    indicator = np.array([0], dtype=np.ubyte)
                # If at least one feature dimension has a value recorded, the indicator bit is set to one
                else:
                    indicator = np.array([1], dtype=np.ubyte)
                # Replace missing values with zero and convert to an array
                value = np.array([v if pd.notna(v) else 0 for v in value], dtype=np.float32)
            else:
                # Feature is not present in the dataframe
                indicator = np.array([0], dtype=np.ubyte)
                value = np.zeros([ft_size], dtype=np.float32)
            feat_indicator_list.append(indicator)
            feat_value_list.append(value)

        return feat_indicator_list, feat_value_list

    def _extract_categorical(
            self,
            data: namedtuple,
            candidate_feats: List[str]
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Extract categorical features from the data.
        
        Args:
            data: Named tuple containing the timeseries data for a single timestep.
            candidate_feats: List of names of candidate features to extract.
        
        Returns:
            Tuple of lists containing indicators and values for each categorical feature.
        """
        feat_indicator_list = []  # A list of all features' indicators for this timestep
        feat_value_list = []  # A list of all features' values for this timestep
        for ft in candidate_feats:
            # Look up the category value map. If the data are already integer indices, work with them as they 
            # are. Otherwise, convert the labels to the corresponding integer indices. The indices need to 
            # start from 1 because zero exclusively represents unrecorded categorical features. If the category
            # map does not start at 1, adjust the category map and, if necessary, the data values.
            category_map = self._get_category_map(ft)
            # Verify whether the map indexes labels from 1 and calculate an adjustment factor
            first_cat_idx = min([int(k) for k in category_map.keys()])
            adjustment = 1 - first_cat_idx  # Zero if no adjustment needed
            if first_cat_idx != 1:
                # Adjust category_map keys to start from 1 and update self.var_properties
                category_map = dict((int(k) + adjustment, v) for k, v in category_map.items())
                self._set_category_map(ft, category_map)
            # Convert the candidate feature to valid Python identifiers that match the new column names
            ft = ft.replace(' ', '_').replace('-', '_')
            # Check if this feature exists in the data
            if ft in data._fields:
                # Get the value for this feature at this timestep
                value = getattr(data, ft)
                if pd.isna(value):
                    # Missing value
                    indicator = np.array([0], dtype=np.ubyte)
                    value = np.array([0], dtype=np.int32)
                else:  # Feature value is recorded
                    if isinstance(value, str):
                        if value not in category_map.values():
                            raise ValueError(
                                f"Category label '{value}' not found in category_map for feature {ft}. "
                                f"Ensure that all category labels are represented in category_map for this "
                                f"feature in variable_properties.yaml."
                            )
                        idx_map = {lab: idx for idx, lab in category_map.items()}
                        value = idx_map[value]  # Convert label to index
                    else:  # Assume value is already an integer index
                        value += adjustment  # `adjustment` = 0 if no adjustment needed
                    indicator = np.array([1], dtype=np.ubyte)
                    value = np.array([value], dtype=np.int32)  # Convert to numpy array
            else:  # Feature not in the DataFrame
                indicator = np.array([0], dtype=np.ubyte)
                value = np.array([0], dtype=np.int32)  # Default value for absent feature
            feat_indicator_list.append(indicator)
            feat_value_list.append(value)
        
        return feat_indicator_list, feat_value_list

    def _extract_text(
        self,
        data: namedtuple,
        candidate_feats: List[str]
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Extract text features from a single timestep in the timeseries data.
        
        Args:
            data: Named tuple containing the timeseries data
            candidate_feats: List of candidate feature names to extract.
        
        Returns:
            Tuple of lists containing indicators and values for each text feature.
        """
        if self.tokenizer is None:
            raise ValueError(
                "Text features are present but no tokenizer is set. Provide a tokenizer in the constructor."
            )
        feat_indicator_list = []
        feat_value_list = []
        feat_mask_list = []  # Will hold attention masks for token sequences of each text feature
        for ft in candidate_feats:
            # Convert the candidate feature to valid Python identifiers that match the new column names
            ft = ft.replace(' ', '_').replace('-', '_')
            if ft in data._fields:
                value = getattr(data, ft)
                if pd.isna(value):
                    indicator = np.array([0], dtype=np.ubyte)
                    value = np.zeros((self.tokenizer.max_length,), dtype=np.int32)  # Placeholder for empty text
                    mask = np.zeros((self.tokenizer.max_length,), dtype=np.ubyte)  # Placeholder for empty attn mask
                else:
                    # Tokenize the text value
                    tokenizer_output = self.tokenizer.process_text(value)
                    tokenized_text = tokenizer_output['input_ids']
                    token_attn_mask = tokenizer_output['attention_mask']
                    indicator = np.array([1], dtype=np.ubyte)
                    value = np.array(tokenized_text, dtype=np.int32)
                    mask = np.array(token_attn_mask, dtype=np.ubyte)
            else:  # Feature not in the DataFrame
                indicator = np.array([0], dtype=np.ubyte)
                value = np.zeros((self.tokenizer.max_length,), dtype=np.int32)  # Placeholder for absent text
                mask = np.zeros((self.tokenizer.max_length,), dtype=np.ubyte)  # Placeholder for absent attention mask
            feat_indicator_list.append(indicator)
            feat_value_list.append(value)
            feat_mask_list.append(mask)

        return feat_indicator_list, feat_value_list, feat_mask_list

    def extract_value_timeseries(
        self,
        data: pd.DataFrame,
        candidate_feats: List[str],
    ) -> ValueAssociatedDataEntry:
        """
        Prepares a value-associated data timeseries for one patient-episode.
        
        Args:
            data: DataFrame with timeseries data from a single patient-episode.
            candidate_feats: List of candidate feature names to extract.
        """

        if data.empty:
            return {}
        
        data = data.copy(deep=True)
        
        # Get lists of candidate features by type
        numeric_features = []
        categorical_features = []
        text_features = []

        for ft in candidate_feats:
            feat_type = self._get_feature_type(ft)
            if feat_type == 'numeric':
                numeric_features.append(ft)
            elif feat_type == 'categorical':
                categorical_features.append(ft)
            elif feat_type == 'text':
                text_features.append(ft)

        if not (numeric_features or categorical_features or text_features):
            warn(
                "No valid feature types found in candidate_feats. Returning empty data. "
                "Ensure that the variable_properties.yaml file contains the correct type information "
                "for each of the features (valid types are 'numeric', 'categorical', or 'text')."
            )
            return {}

        # Prepare dicts for each feature type
        val_data = {}
        # Initialize dict items for feature types that are present in the data
        val_data['numeric'] = {'indicators': [], 'values': []}
        val_data['categorical'] = {'indicators': [], 'values': []}
        # Text gets attention masks for length-padded token sequences -- not the same as timestep padding masks
        val_data['text'] = {'indicators': [], 'values': [], 'masks': []}

        if text_features:
            if self.tokenizer is None:
                raise ValueError(
                    "Text features are present but no tokenizer is set. Provide a tokenizer in the constructor."
                )

        val_data['times'] = []  # Fill with scalar timestamp arrays
        val_data['masks'] = []  # Fill with scalar mask arrays (1 for part of timeseries, 0 for masked out)

        # Convert column names to valid Python identifiers that are compatible with namedtuples
        data.columns = [col.replace(' ', '_').replace('-', '_') for col in data.columns]

        # Iterate over rows (timesteps) in the DataFrame and extract each feature type
        for record in data.itertuples(index=True, name='PatientRecord'):
            timestamp = record.Index / np.timedelta64(1, 'h')  # Convert to hours relative to start of episode
            val_data['times'].append(np.array([timestamp], dtype=np.float32))  # Store as scalar array
            val_data['masks'].append(np.array([1], dtype=np.ubyte))  # Mask is 1 for records in timeseries, 0 if masked
            
            feat_indicator_list, feat_value_list = self._extract_numeric(record, numeric_features)
            val_data['numeric']['indicators'].append(feat_indicator_list)
            val_data['numeric']['values'].append(feat_value_list)

            feat_indicator_list, feat_value_list = self._extract_categorical(record, categorical_features)
            val_data['categorical']['indicators'].append(feat_indicator_list)
            val_data['categorical']['values'].append(feat_value_list)

            feat_indicator_list, feat_value_list, feat_mask_list = self._extract_text(record, text_features)
            val_data['text']['indicators'].append(feat_indicator_list)
            val_data['text']['values'].append(feat_value_list)
            val_data['text']['masks'].append(feat_mask_list)

        # Pad the data to the maximum timeseries length. Memory-intensive, but necessary for batching
        ts_length = len(val_data['times'])
        if ts_length < self.max_timeseries_length:
            pad_length = self.max_timeseries_length - ts_length
            for data_type in ['numeric', 'categorical', 'text']:
                for key in ['indicators', 'values', 'masks']:
                        # Select the appropriate dtype for the arrays
                        if key == 'indicators':
                            arr_dtype = np.ubyte
                        elif key == 'values':
                            arr_dtype = np.float32 if data_type == 'numeric' else np.int32
                        elif key == 'masks':
                            if data_type == 'text':
                                arr_dtype = np.ubyte
                            else:
                                continue  # The 'masks' key is only present for text features
                        # Pad with lists of like-sized zero arrays to max_timeseries_length
                        n_features = len(val_data[data_type][key][-1])
                        padding = [np.zeros_like(val_data[data_type][key][-1][i], dtype=arr_dtype)
                                   for i in range(n_features)]
                        # Extend the list with references -- NOT copies -- to the padding zero arrays
                        # e.g. `padding[0] is padding[1]` evaluates to True
                        padding = [padding for _ in range(pad_length)]
                        val_data[data_type][key].extend(padding)
            time_padding = np.zeros((1, ), dtype=np.float32)
            mask_padding = np.zeros((1, ), dtype=np.ubyte)
            val_data['times'].extend([time_padding for _ in range(pad_length)])  # Not copies
            val_data['masks'].extend([mask_padding for _ in range(pad_length)])  # Not copies

        return val_data

    def extract_event_timeseries(
        self,
        data: pd.DataFrame,
        candidate_feats: List[str]
    ) -> EventAssociatedDataEntry:
        """
        Extract event-associated timeseries data.
        
        Events maintain their original timestamps without resampling.
        """

        data = data.copy(deep=True)
        
        if data.empty:
            return {}
        
        event_data = {}

        event_data['indicators'] = []  # Fill with scalar event indicator arrays for each feature
        event_data['times'] = []  # Fill with scalar timestamp arrays
        event_data['masks'] = []  # Fill with scalar mask arrays (1 for part of timeseries, 0 for masked out)

        # Convert column names to valid Python identifiers that are compatible with namedtuples
        data.columns = [col.replace(' ', '_').replace('-', '_') for col in data.columns]
        
        # Iterate over rows (timesteps) in the DataFrame and extract each feature type
        for record in data.itertuples(index=True, name='PatientRecord'):
            feat_indicator_list = []  # A list of all features' indicators for this timestep
            timestamp = record.Index / np.timedelta64(1, 'h')
            event_data['times'].append(np.array([timestamp], dtype=np.float32))  # Store as scalar array
            event_data['masks'].append(np.array([1], dtype=np.ubyte))  # Mask is 1 for all recorded events
            for ft in candidate_feats:
                ft_size = self._get_feature_size(ft)
                # Convert the candidate feature to valid Python identifiers that match the new column names
                ft = ft.replace(' ', '_').replace('-', '_')
                cols = sorted(self._get_feature_column_names(ft, data))
                if cols:
                    if ft_size != len(cols):
                        raise ValueError(
                            f"The number of columns for '{ft}' does not match its expected dimension ({ft_size})"
                        )
                # Check whether there is at least one recorded value for a component of the feature
                # This is to cover vector-valued features where some components may be missing.
                value = [getattr(record, col) for col in cols]
                # If the feature value is missing in all dims the indicator bit is set to zero
                if all([(pd.isna(v) or v == '') for v in value]):
                    indicator = np.array([0], dtype=np.ubyte)
                # If at least one feature dimension has a value recorded, the indicator bit is set to one
                else:
                    indicator = np.array([1], dtype=np.ubyte)
                # Append the indicator to the list for this timestep
                feat_indicator_list.append(indicator)
            # Append the feature indicator for this timestamp to the event data
            event_data['indicators'].append(feat_indicator_list)

        # Pad the data to the maximum timeseries length. Memory-intensive, but necessary for batching
        ts_length = len(event_data['times'])
        if ts_length < self.max_timeseries_length:
            pad_length = self.max_timeseries_length - ts_length
            n_features = len(event_data['indicators'][-1])
            event_padding = np.zeros((1, ), dtype=np.ubyte)
            # Extend with references -- NOT copies -- of the padding array
            padding = [[event_padding for _ in range(n_features)] for _ in range(pad_length)]
            event_data['indicators'].extend(padding)
            # Pad the timestamps and masks with references to zero arrays
            time_padding = np.zeros((1, ), dtype=np.float32)
            mask_padding = np.zeros((1, ), dtype=np.ubyte)
            event_data['times'].extend([time_padding for _ in range(pad_length)])
            event_data['masks'].extend([mask_padding for _ in range(pad_length)])
        
        return event_data  

    def extract_static_data(
        self,
        data: Union[pd.Series, pd.DataFrame],
        candidate_feats: List[str]
    ) -> StaticDataEntry:
        """
        Extract static data (single values per patient).
        """
        if data.empty:
            return []
        
        # Convert to namedtuple only if data is a pandas Series
        if isinstance(data, pd.Series):
            data = namedtuple('StaticData', data.index)(*data)
        elif isinstance(data, pd.DataFrame):
            data = namedtuple('StaticData', data.columns)(*data.iloc[0])
        else:
            raise TypeError(f'Unsupported data type: {type(data)}. Expected pd.Series or pd.DataFrame.')
        
        static_data = []
        
        # Process each feature according to its type
        for ft in candidate_feats:
            feat_type = self._get_feature_type(ft)
            if feat_type == 'numeric':
                _, value = self._extract_numeric(data, [ft])
                static_data.extend(value)
            elif feat_type == 'categorical':
                _, value = self._extract_categorical(data, [ft])
                static_data.extend(value)
            elif feat_type == 'text':
                _, value = self._extract_text(data, [ft])
                static_data.extend(value)
        
        return static_data


def standardize_feats(
    x: List[ValueAssociatedDataEntry], 
    save_path: Optional[str] = None, 
    load_path: Optional[str] = None
) -> List[ValueAssociatedDataEntry]:
    """Scale and center the non-zero values of features using their mean and the 5th-95th percentile range.

    This function standardizes the non-zero values of the input array `x`. If `load_path` is provided, the function 
    loads the 5th and 95th percentiles and means of feature values from a .npz file and uses the them for the 
    standardization. If `load_path` is not provided, the function calculates the means and percentiles of each feature 
    from its non-zero values across all samples and timesteps. If `save_path` is provided, the calculated percentiles 
    and means are saved to a .npz file. The function returns the input with standardized values.

    Args:
        x (List[ValueAssociatedDataEntry]): A list of dictionaries containing feature value data. A ValueAssocatedDataEntry is a dictionary with keys that correspond to different data types (numeric, categorical, text) as well as timestamps for each record. Each of those keys maps to a dictionary with keys for feature indicators and values. Each item is a list of lists of numpy arrays where the outer list corresponds to timestamps and the innter list corresponds to features. This function only standardizes the numeric features.
        save_path (str, optional): The path to save the calculated percentiles and means to a .npz file.
        load_path (str, optional): The path to load the calculated percentiles and means from a .npz file.

    Returns:
        List[ValueAssociatedDataEntry]: The input list of ValueAssociatedDataEntry dictionaries with standardized numeric feature values.
    """

    if load_path is not None:
        # Load the 5th, 95th percentiles and means of feature values
        data = np.load(load_path) 
        p5 = data['p5']
        p95 = data['p95']
        means = data['means']
    else:
        num_ind_data = []  # EpisodeList[TimestepList[FeatList[np.ndarray]]]
        num_val_data = []  # EpisodeList[TimestepList[FeatList[np.ndarray]]]
        for episode in x:
            num_ind_data.append(episode['numeric']['indicators'])
            num_val_data.append(episode['numeric']['values'])

        n_features = len(num_val_data[0][0])  # Number of features is the same for all records
        
        # Pre-allocate arrays to avoid memory fragmentation
        means = np.zeros(n_features)
        p5 = np.zeros(n_features)
        p95 = np.zeros(n_features)
        
        # Process one feature at a time to minimize memory usage
        for i in range(n_features):
            obs_values = []  # Store (feat_dim, ) value arrays for feature i across all episodes and timesteps
            # Iterate over patient-episodes
            for episode_indicators, episode_values in zip(num_ind_data, num_val_data):
                # Iterate over individual timesteps in the episode
                for record_indicators, record_values in zip(episode_indicators, episode_values):
                    if record_indicators[i] == 1:  # Only append values of features that were actually observed
                        obs_values.append(record_values[i])
            
            if obs_values:
                # Stack and compute statistics immediately, then discard the array
                obs_array = np.stack(obs_values, axis=0)
                means[i] = np.mean(obs_array)
                norms = np.linalg.norm(obs_array, ord=2, axis=0 if obs_array.ndim > 1 else None)
                if np.isscalar(norms):
                    norms = np.array([norms])
                p5[i] = np.percentile(norms, 5)
                p95[i] = np.percentile(norms, 95)
                # Free memory immediately
                del obs_array, obs_values, norms
            else:
                means[i] = 0
                p5[i] = 0
                p95[i] = 0

    if save_path is not None:
        # Save arrays of feature-wise percentiles and means to a .npz file
        np.savez(save_path, p5=p5, p95=p95, means=means)

    # Center the original feature values on the mean and scale by the 5th-95th %ile range
    for i, episode in enumerate(x):
        for j, timestep in enumerate(episode['numeric']['values']):
            for k, feature in enumerate(timestep):
                if p5[k] == p95[k]:
                    # If the 5th and 95th percentiles are equal, set the feature value to 0
                    episode['numeric']['values'][j][k] = np.zeros_like(feature)
                else:
                    # Standardize the feature by centering on the mean and scaling by the 5th-95th %ile range
                    std_vals = (feature - means[k]) / (p95[k] - p5[k])
                    x[i]['numeric']['values'][j][k] = std_vals

    return x


def filter_timeseries_records(
        numeric_data: pd.DataFrame,
        event_data: pd.DataFrame,
        text_data: Optional[pd.DataFrame] = None,
        max_history_len: int = 0,
        max_episode_len: int = 100,
        max_episode_len_hours: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    def filter(df):
       
        df = df.sort_index()
            
        # Exclude records collected more than x hours into the current ICU stay episode
        if max_episode_len_hours is not None:
            selected_records = df.index < np.timedelta64(max_episode_len_hours, 'h')
            df = df.loc[selected_records, :]

        # Get indices of up to x records from the current ICU stay episode, starting from the earliest record
        episode_record_indices = np.where(df.index >= np.timedelta64(0, 'h'))[0]
        episode_len = min(len(episode_record_indices), max_episode_len)
        episode_record_indices = episode_record_indices[:episode_len]  # Indices of the stay's earliest records
        
        # Get indices of up to x most recent records that were collected before the current ICU stay episode
        historic_record_indices = np.where(df.index < np.timedelta64(0, 'h'))[0]
        history_len = min(len(historic_record_indices), max_history_len)
        historic_record_indices = historic_record_indices[-history_len:]  # Indices of most recent historic records
        
        # Combine to give the indices of records to keep
        keep = np.concatenate((historic_record_indices, episode_record_indices))  # Indices of records to keep
        
        return df.iloc[keep, :]

    event_data = filter(event_data)
    
    # Text data will eventually be embedded and merged with numeric feature data, so they must be filtered together
    # to ensure that the length of the merged timeseries does not exceed the maximum allowed timeseries length.
    if text_data is not None:
        # Merge text and numeric data on timestamps; used for filtering the numeric and text data
        merged = numeric_data.merge(text_data, how='outer', left_index=True, right_index=True, indicator=True)
        # Get the names of the numeric and text features so that they can be recovered after filtering
        numeric_feats = numeric_data.columns
        text_feats = text_data.columns
        filtered = filter(merged)
        # Split the filtered numeric and text data back into separate DataFrames
        numeric_data = filtered.loc[filtered['_merge'].isin(['both', 'left_only']), numeric_feats]
        text_data = filtered.loc[filtered['_merge'].isin(['both', 'right_only']), text_feats]
    else:
        # No text data, so there's no need to merge timesteps
        numeric_data = filter(numeric_data)

    return numeric_data, event_data, text_data


def _init_worker(max_timeseries_length: int):
    """Initialize worker process with a DataProcessor instance."""
    global _worker_processor
    _worker_processor = DataProcessor(max_timeseries_length, tokenizer=LlamaTextProcessor())


def _process_single_episode(
    i: int,
    reader: MIMICDataReader,
    max_history_len_steps: int,
    max_episode_len_steps: int,
    max_episode_len_hours: Optional[int],
    min_episode_len_steps: Optional[int],
    min_episode_len_hours: Optional[int]
) -> Optional[Tuple[int, dict, dict, list, dict]]:
    """Process a single patient episode using the worker's processor."""
    
    global _worker_processor
    processor = _worker_processor
    
    try:
        _, statics, val_data, event_data, text_data, targets = reader[i]
        targets = dict(zip(['mortality', 'length_of_stay', 'phenotype'], [np.array(t) for t in targets]))
        
        # ...existing filtering and processing logic...
        
        # Skip episodes with length of stay < minimum number of hours
        if min_episode_len_hours is not None:
            if targets['length_of_stay'] < min_episode_len_hours:
                return None

        # Skip episodes with fewer than minimum records
        if min_episode_len_steps is not None:
            min_timestamp = np.timedelta64(0, 'h')
            if max_episode_len_hours is not None:
                max_timestamp = np.timedelta64(max_episode_len_hours, 'h')
                if text_data is not None:
                    merged = val_data.merge(text_data, how='outer', left_index=True, right_index=True)
                    is_current_record = (min_timestamp <= merged.index) & (merged.index < max_timestamp)
                else:
                    is_current_record = (min_timestamp <= val_data.index) & (val_data.index < max_timestamp)
            else:
                if text_data is not None:
                    merged = val_data.merge(text_data, how='outer', left_index=True, right_index=True)
                    is_current_record = (min_timestamp <= merged.index)
                else:
                    is_current_record = (min_timestamp <= val_data.index)
            n_current_records = is_current_record.sum()
            if n_current_records < min_episode_len_steps:
                return None
        
        # Resample value-associated data
        val_data = val_data.set_index(val_data.index.ceil('h')).resample('1h', closed='right', label='right').mean()
        val_data = val_data.dropna(axis=0, how='all')

        val_data, event_data, text_data = filter_timeseries_records(
            val_data, event_data, text_data, max_history_len_steps, max_episode_len_steps, max_episode_len_hours
        )

        if text_data is not None:
            val_data = val_data.merge(text_data, how='outer', left_index=True, right_index=True)
            val_data.columns = [col.rsplit('_', 1)[0] if col.endswith(('_left', '_right')) else col 
                                for col in val_data.columns]

        val_data = processor('value', val_data, reader.valued_feats + reader.text_feats)
        event_data = processor('event', event_data, reader.event_feats)
        static_data = processor('static', statics, reader.static_feats)

        # Normalize length of stay
        if max_episode_len_hours is not None:
            targets['length_of_stay'] = targets['length_of_stay'] - max_episode_len_hours
        else:
            max_val_ts_timestamp = max(val_data['times'])
            max_event_ts_timestamp = max(event_data['times'])
            max_observed_timestamp = max(max_val_ts_timestamp, max_event_ts_timestamp)
            targets['length_of_stay'] = targets['length_of_stay'] - max_observed_timestamp

        return (i, val_data, event_data, static_data, targets)
        
    except Exception as e:
        print(f"Error processing episode {i}: {e}")
        return None


def extract_mimic(
        reader: MIMICDataReader, 
        suffix: str,
        output_dir: str,
        max_episode_len_steps: int, 
        max_history_len_steps: int = 0,
        min_episode_len_steps: Optional[int] = 10,
        min_episode_len_hours: Optional[int] = 48,
        max_episode_len_hours: Optional[int] = 48,
        n_workers: Optional[int] = None
) -> None:
    """
    Reads MIMIC ICU stay timeseries data from CSV files and pickles it in a format compatible with downstream predictive models. The pickled object is a dictionary with five keys: 'id', which is a list of patient-episode IDs; 'val_data', which stores value-associated data; 'event_data', which stores event-associated data; 'static_data', which stores time-invariant patient parameters; and 'targets', which stores in-hospital mortality, length of stay, and phenotype data for downstream prediction. This function trims the data contained in the CSV files to the desired records, extracts them, and standardizes the values of value-associated data (see the `standardize_feats` function) before dumping the restructured data to disk.

    Args:

        suffix (str): The data partition for which to extract data. Accepts 'train', 'val', or 'test'.
        output_dir (str): The directory where the pickled data will be dumped. Saved as `{suffix}.pkl`.
        max_episode_len_steps (int): The maximum number of records (timesteps) to include from within each ICU episode. 
            These records are counted starting from the time of admission to ICU.
        max_history_len_steps: The maximum number of records to include that predate admission to the ICU. These 
            records are counted backwards starting from the time of admission to ICU. Defaults to 0.
        min_episode_len_steps (int, optional): The minimum number of records that must exist in an ICU stay within the 
            timeframe established by `max_episode_len_hours` if it was provided. If the number of records collected during that time is less than `min_episode_len_steps`, the episode is ignored. Defaults to 10. If None, there is no restriction on the number of records that must be present.
        min_episode_episode_len_hours (int, optional): The minimum duration of an ICU stay, in hours, for it to be 
            included in the dataset. Defaults to 48. If None, there is no restriction on the minimum length of ICU stay.
        max_episode_len_hours (int, optional): The latest timestamp, in hours, which can be included in the extracted 
            data. For example, if `max_episode_len_hours` is set to 48 (the default), only the records collected within the first 48 hours of the ICU stay are included. If None, all records will be included subject to the other restrictions.
        n_workers (int, optional): The number of parallel worker processes to use for data extraction. If None, 
            defaults to 1.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if reader.prediction_task != 'all':
        raise ValueError(f'reader.prediction_task: Expected "all", got {reader.prediction_task}')

    total_episodes = len(reader.patient_episode_ids)
    max_ts_len = max_history_len_steps + max_episode_len_steps

    # Set number of workers
    if n_workers is None:
        n_workers = 1
    
    print(f"Processing {total_episodes} episodes using {n_workers} workers...")

    # Create partial function with fixed arguments
    process_fn = partial(
        _process_single_episode,
        reader=reader,
        max_history_len_steps=max_history_len_steps,
        max_episode_len_steps=max_episode_len_steps,
        max_episode_len_hours=max_episode_len_hours,
        min_episode_len_steps=min_episode_len_steps,
        min_episode_len_hours=min_episode_len_hours
    )

    # Process episodes in parallel with per-worker processor initialization
    all_val_data = []
    all_event_data = []
    all_static_data = []
    all_target_data = []
    ids = []
    n_episodes_ignored = 0
    
    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(max_ts_len,)) as pool:
        for result in tqdm(
            pool.imap(process_fn, range(total_episodes), chunksize=10),
            total=total_episodes,
            desc=f"Extracting {suffix} patient records from {reader.data_root_path}"
        ):
            if result is None:
                n_episodes_ignored += 1
            else:
                i, val_data, event_data, static_data, targets = result
                ids.append(i)
                all_val_data.append(val_data)
                all_event_data.append(event_data)
                all_static_data.append(static_data)
                all_target_data.append(targets)
    print(f"Extracted records from {total_episodes-n_episodes_ignored} ICU stay episodes, ignored {n_episodes_ignored} "
          f"episodes that didn't meet filtering criteria.")
    sys.stdout.flush()

    # Restrict the data to the patient-episode IDs that survived filtering
    patient_episode_ids = np.array(reader.patient_episode_ids)[ids]
    patient_episode_ids = patient_episode_ids.tolist()
    
    # Standardize the numeric value-associated data
    # NOTE: Xu et al. standardized the training, validation, and test set data each with their own summary statistics,
    # but instead I use the summary statistics from the training data for everything. This makes more sense because
    # during inference with a deployed model we won't necessarily have summary statistics for new data, and we assume
    # that the unseen data are sampled from the same distriubtion as the training data anyway. Even if we do have the
    # summary statistics for the unseen data, the training set is larger and better approximates the true distribution.
    summary_statistic_path = os.path.join(output_dir, 'summary_statistics_train.npz')
    if suffix == 'train':
        print(f"Calculating smmary statistics...", flush=True)
        sys.stdout.flush()
        # Calculate summary statistics for the training set data and write to disk
        all_val_data = standardize_feats(all_val_data, save_path=summary_statistic_path)
    else:
        if not os.path.exists(summary_statistic_path):
            raise FileNotFoundError(
                'Validation and test set data are standardized with summary statistics calculated from the training '
                'set data, but summary_statistics_train.npz was not found. Please run the training data extraction '
                'first to generate the summary statistics.'
            )
        print(f"Loading and applying summary statistics...", flush=True)
        sys.stdout.flush()
        # Standardize the validation and test set data using the summary statistics calculated from the training set
        all_val_data = standardize_feats(all_val_data, load_path=summary_statistic_path)

    # Write to disk
    file_out = os.path.join(output_dir, f'{suffix}.pkl')

    with open(file_out, 'wb') as f_out:
        patient_episodes = {
            'id': patient_episode_ids,
            'val_data': all_val_data,
            'event_data': all_event_data,
            'static_data': all_static_data,
            'targets': all_target_data
        }
        pickle.dump(patient_episodes, f_out)
    print(f"Extracted {suffix} data written to {file_out}\n")


def prepare_dataloaders(
    data_dir: str,
    batch_size: int,
    pretrain_ratio: Optional[float] = None,
    collate_fn: Optional[callable] = None
) -> List[DataLoader]:
    """Prepare training, validation, and test dataloaders for the MIMIC dataset.
    
    This function prepares training, validation, and test dataloaders for the MIMIC dataset. The function loads the
    .pkl files containing the extracted data, constructs PyTorch Dataset instances from the data, and creates DataLoader
    instances for each data partition. The function returns the training, validation, and test Dataloaders in a list.

    Args:
        data_dir (str): The directory containing the .pkl files extracted from the MIMIC dataset.
        batch_size (int): The batch size for the DataLoader instances.
        pretrain_ratio (float, optional): The ratio of the training data to use for pretraining. If `None`, no 
            pretraining is performed.
        collate_fn (callable, optional): A function to merge a list of samples to form a mini-batch. If `None`, the 
            default PyTorch collate function is used.
    
    Returns:
        List[DataLoader]: A list containing the training, validation, and test Dataloaders.
    
    """

    datasets = {}

    for partition in ['train', 'val', 'test']:
        # If validation data is not available, set the validation dataset to None.
        # In this case, there will be no validation dataloader in the dataloader_list output from this function.
        # If training or test data are not available, raise an exception.
        data_file = f'{data_dir}/{partition}.pkl'
        if not os.path.exists(data_file):
            if partition != 'val':
                raise FileNotFoundError(f'{partition}.pkl not found in {data_dir}.')
            else:
                datasets['val'] = None
                continue
        
        with open(data_file, 'rb') as f_in:
            patient_episodes = pickle.load(f_in)

        ds = MixedDataset(
            id=patient_episodes['id'],
            val_data=patient_episodes['val_data'],
            event_data=patient_episodes['event_data'],
            static_data=patient_episodes['static_data'],
            targets=patient_episodes['targets']
        )

        datasets[partition] = ds

    datasets_list = []

    if pretrain_ratio is not None:
        pretrain_size = int(pretrain_ratio * len(datasets_list['train']))  # Floor
        pretrain_dataset, train_dataset = train_test_split(datasets['train'], train_size=pretrain_size, random_state=42)
        datasets_list.append(pretrain_dataset)
        datasets_list.append(train_dataset)
    else:
        datasets_list.append(datasets['train'])
    
    if datasets['val'] is not None:
        datasets_list.append(datasets['val'])
    
    datasets_list.append(datasets['test'])

    dataloader_list = []
    for ds in datasets_list:
        dataloader_list.append(
            DataLoader(
                ds, 
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
        )

    return dataloader_list


def prepare_dataloaders_hdf5(
    data_dir: str,
    batch_size: int,
    preload: bool = True,
    pretrain_ratio: Optional[float] = None,
    collate_fn: Optional[callable] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2
) -> List:
    """
    Prepare training, validation, and test dataloaders using HDF5 datasets.
    
    This is a drop-in replacement for prepare_dataloaders that uses HDF5 files instead of pickle files, which is far more efficient for large datasets.
    
    Args:
        data_dir (str): Directory containing .h5 files
        batch_size (int): Batch size for DataLoader
        preload (bool): If True (default), load all data into RAM at initialization. Much faster than pickle and 
            eliminates per-batch I/O. Set to False only if RAM is limited.
        pretrain_ratio (float): Ratio of training data for pretraining (not yet supported)
        collate_fn (callable): Custom collate function
        num_workers (int): Number of worker processes for data loading. With preload=True, workers share data via 
            copy-on-write after fork.
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        prefetch_factor (int): Number of batches to prefetch per worker
        
    Returns:
        List of DataLoader instances
    """
    from torch.utils.data import DataLoader
    
    if pretrain_ratio is not None:
        raise NotImplementedError(
            "pretrain_ratio is not yet supported with HDF5 datasets. "
            "Consider creating separate pretrain/train HDF5 files."
        )
    
    datasets = {}
    
    for partition in ['train', 'val', 'test']:
        h5_path = os.path.join(data_dir, f'{partition}.h5')
        
        if not os.path.exists(h5_path):
            if partition == 'val':
                datasets['val'] = None
                continue
            else:
                raise FileNotFoundError(f'{partition}.h5 not found in {data_dir}')
        
        datasets[partition] = HDF5Dataset(h5_path, preload=preload)
    
    datasets_list = [datasets['train']]
    
    if datasets['val'] is not None:
        datasets_list.append(datasets['val'])
    
    datasets_list.append(datasets['test'])
    
    dataloader_list = []
    for ds in datasets_list:
        dataloader_list.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=num_workers > 0
            )
        )
    
    return dataloader_list


def _cast_input_tensors(res_dict: dict) -> dict:
    """
    Cast tensors to appropriate types (float, long, etc.) based on their semantic meaning.
    
    Args:
        res_dict: Dictionary containing nested tensors
        
    Returns:
        Dictionary with tensors cast to appropriate types
    """
    for key, value in res_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    # Handle 3rd level nesting (val_data -> numeric/categorical/text -> indicators/values/masks)
                    for subsubkey, subsubvalue in subvalue.items():
                        if isinstance(subsubvalue, torch.Tensor):
                            # Special handling for text data - keep token IDs and masks as integers
                            if subkey == 'text' and subsubkey in ['values', 'masks']:
                                res_dict[key][subkey][subsubkey] = subsubvalue.long()
                            # Special handling for categorical data - keep values as integers
                            elif subkey == 'categorical' and subsubkey == 'values':
                                res_dict[key][subkey][subsubkey] = subsubvalue.long()
                            else:
                                res_dict[key][subkey][subsubkey] = subsubvalue.float()
                        elif isinstance(subsubvalue, list):
                            # Handle lists of tensors
                            if subkey == 'text' and subsubkey in ['values', 'masks']:
                                res_dict[key][subkey][subsubkey] = [t.long() for t in subsubvalue if isinstance(t, torch.Tensor)]
                            elif subkey == 'categorical' and subsubkey == 'values':
                                res_dict[key][subkey][subsubkey] = [t.long() for t in subsubvalue if isinstance(t, torch.Tensor)]
                            else:
                                res_dict[key][subkey][subsubkey] = [t.float() for t in subsubvalue if isinstance(t, torch.Tensor)]
                elif isinstance(subvalue, torch.Tensor):
                    # Handle targets - mortality and phenotype should remain float, length_of_stay can be float
                    res_dict[key][subkey] = subvalue.float()
                elif isinstance(subvalue, list):
                    res_dict[key][subkey] = [t.float() for t in subvalue if isinstance(t, torch.Tensor)]
        elif isinstance(value, torch.Tensor):
            res_dict[key] = value.float()
    
    return res_dict


def _input_tensors_to_device(res_dict: dict, device: Union[torch.device, str]) -> dict:
    """
    Move all tensors in the nested dictionary to the specified device.
    
    Args:
        res_dict: Dictionary containing nested tensors
        device: Target device for tensors
        
    Returns:
        Dictionary with all tensors moved to the specified device
    """
    for key, value in res_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    # Handle 3rd level nesting (val_data -> numeric/categorical/text -> indicators/values/masks)
                    for subsubkey, subsubvalue in subvalue.items():
                        if isinstance(subsubvalue, torch.Tensor):
                            res_dict[key][subkey][subsubkey] = subsubvalue.to(device)
                        elif isinstance(subsubvalue, list):
                            # Handle lists of tensors
                            res_dict[key][subkey][subsubkey] = [t.to(device) for t in subsubvalue if isinstance(t, torch.Tensor)]
                elif isinstance(subvalue, torch.Tensor):
                    res_dict[key][subkey] = subvalue.to(device)
                elif isinstance(subvalue, list):
                    res_dict[key][subkey] = [t.to(device) for t in subvalue if isinstance(t, torch.Tensor)]
        elif isinstance(value, torch.Tensor):
            res_dict[key] = value.to(device)
    
    return res_dict


def prepare_input_tensors(batch: MixedDataset, device) -> MixedTensorDataset:

    # Final shapes are stated in comments
    res_dict = {
        'val_data': {
            'numeric': {
                'indicators': [],  # (batch_size, max_ts_len, n_numeric_features) tensor
                'values': []  # List of (batch_size, max_ts_len, feature_dim) tensors x n_numeric_features
            },
            'categorical': {
                'indicators': [],  # (batch_size, max_ts_len, n_categorical_features) tensor
                'values': []  # List of (batch_size, max_ts_len, feature_dim) tensors x n_categorical_features
            },
            'text': {
                'indicators': [],  # (batch_size, max_ts_len, n_text_features) tensor
                'values': [],  # List of (batch_size, max_ts_len, feature_dim) tensors x n_text_features
                'masks': []  # List of (batch_size, max_ts_len, feature_dim) tensors x n_text_features
            },
            'times': [],  # (batch_size, max_ts_len) tensor
            'masks': []  # (batch_size, max_ts_len) tensor
        },
        'event_data': {
            'indicators': [],  # (batch_size, max_ts_len, n_event_features) tensor
            'times': [],  # (batch_size, max_ts_len) tensor
            'masks': []  # (batch_size, max_ts_len) tensor
        },
        'static_data': [], # (batch_size, total_static_feature_dim) tensor
        'targets': {
            'mortality': [],  # (batch_size, 1) tensor
            'length_of_stay': [],  # (batch_size, 1) tensor
            'phenotype': []  # (batch_size, n_phenotypes) tensor
        }
    }

    # Prepare the value-associated data
    max_ts_len = len(batch['val_data']['times'])  # Max timeseries length for value-associated data
    for key in res_dict['val_data'].keys():
        data = batch['val_data'][key]
        if key in ['numeric', 'categorical', 'text']:
            n_feats = len(data['indicators'][0])  # Number of features in the value-associated data
            ind_data = data['indicators']
            val_data = data['values']
            msk_data = data.get('masks', None)  # Masks are only present for text features
            for f in range(n_feats):
                # Stack the feature tensors across timesteps
                # ind_tnsr shape: (batch_size, max_ts_len, 1)
                # val_tnsr shape: (batch_size, max_ts_len, feature_dim)
                ind_tnsr = torch.stack([ind_data[t][f] for t in range(max_ts_len)], dim=1)
                val_tnsr = torch.stack([val_data[t][f] for t in range(max_ts_len)], dim=1)
                res_dict['val_data'][key]['indicators'].append(ind_tnsr)  # Still needs to be concatenated
                res_dict['val_data'][key]['values'].append(val_tnsr)
                if msk_data is not None:
                    msk_tnsr = torch.stack([msk_data[t][f] for t in range(max_ts_len)], dim=1)
                    res_dict['val_data'][key]['masks'].append(msk_tnsr)
            # Concatenate the feature indicator tensors across features
            # resulting shape: (batch_size, max_ts_len, n_features)
            if res_dict['val_data'][key]['indicators']:
                res_dict['val_data'][key]['indicators'] = torch.cat(res_dict['val_data'][key]['indicators'], dim=2)
            else:
                batch_size = batch['val_data']['times'][0].shape[0]
                # Create a dummy empty tensor if there are no features of this type
                res_dict['val_data'][key]['indicators'] = torch.empty((batch_size, max_ts_len, 0))
        elif key in ['times', 'masks']:
            res_dict['val_data'][key] = torch.cat(data, dim=1)  # Shape: (batch_size, max_ts_len)

    # Prepare the event-associated data
    max_ts_len = len(batch['event_data']['times'])  # Max timeseries length for event-associated data
    n_feats = len(batch['event_data']['indicators'][0])  # Number of event-associated features
    for f in range(n_feats):
        # Stack the feature tensors across timesteps
        # ind_tnsr shape: (batch_size, max_ts_len, 1)
        ind_data = batch['event_data']['indicators']
        ind_tnsr = torch.stack([ind_data[t][f] for t in range(max_ts_len)], dim=1)
        res_dict['event_data']['indicators'].append(ind_tnsr)
    # Concatenate the feature indicator tensors across features
    # resulting shape: (batch_size, max_ts_len, n_event_features)
    if res_dict['event_data']['indicators']:
        res_dict['event_data']['indicators'] = torch.cat(res_dict['event_data']['indicators'], dim=2)
    else:
        batch_size = batch['event_data']['times'][0].shape[0]
        # Create a dummy empty tensor if there are no event-associated features
        res_dict['event_data']['indicators'] = torch.empty((batch_size, max_ts_len, 0))
    res_dict['event_data']['times'] = torch.cat(batch['event_data']['times'], dim=1)  # Shape: (batch_size, max_ts_len)
    res_dict['event_data']['masks'] = torch.cat(batch['event_data']['masks'], dim=1)  # Shape: (batch_size, max_ts_len)

    # Prepare the static data
    res_dict['static_data'] = torch.cat(batch['static_data'], dim=1)  # Shape: (batch_size, total_feature_dim)

    # Prepare the target data
    trgt_data = batch['targets']
    res_dict['targets']['mortality'] = trgt_data['mortality'].reshape(-1, 1)  # Shape: (batch_size, 1)
    res_dict['targets']['length_of_stay'] = trgt_data['length_of_stay'].reshape(-1, 1)  # Shape: (batch_size, 1)
    res_dict['targets']['phenotype'] = trgt_data['phenotype']  # Shape: (batch_size, n_phenotypes)

    # Cast tensors to appropriate types and move to the specified device
    res_dict = _cast_input_tensors(res_dict)
    if device is not None:
        res_dict = _input_tensors_to_device(res_dict, device)
            
    return res_dict


def collate_as_tensors(
    batch: List[tuple],
    device: Optional[Union[torch.device, str]] = None
) -> MixedTensorDataset:
    
    """Device-aware collation of a batch of instances from a MixedDataset.
    
    This function collates a batch of instances from a `MixedDataset` into a `MixedTensorDataset` and moves it to the specified device.

    Args:
        batch (List[tuple]): A list of instances from `MixedDataset.__getitem__()`, where each tuple contains
            `(val_data, event_data, static_data, targets)`.
        device: Target device for the collated batch. If `None`, the batch is kept on CPU.
    
    """

    batch = default_collate(batch)
    prepared_batch = prepare_input_tensors(batch, device)

    return prepared_batch
