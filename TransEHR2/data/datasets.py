import h5py
import numpy as np

from typing import Any, Dict, List



class MixedDataset(object):
    """A dataset for input to a TransEHR2 model.
    
    The dataset is a list of patient-episodes where data from each episode are contained 
    in a nested dictionary with the following structure:

    * *id* (int): The patient-episode ID.

    * *val_data* (Dict): A dictionary containing value-associated data that will be used as 
      input to the ELECTRA-style generator-discriminator networks.
      
      * *numeric* (Dict): Contains real-valued feature data.
        * *indicators* (List[List[np.ndarray]]): A timesteps -> features nested list of scalar 
          arrays indicating whether the feature was recorded at that timestep. Defaults to 
          an array of zeros at masked timesteps.
        * *values* (List[List[np.ndarray]]): A timesteps -> features nested list of arrays 
          containing the actual values recorded at that timestep. The length of the arrays 
          may vary, but should be consistent for each feature across timesteps and episodes.
          Arrays default to zeros if the feature was not recorded at that timestep.

      * *categorical* (Dict): Contains categorical feature data.
        * *indicators* (List[List[np.ndarray]]): see above.
        * *values* (List[List[np.ndarray]]): Values should be scalar arrays of category indices, 
          with zero reserved to indicate that the categorical feature was not recorded at a 
          particular timestep.

      * *text* (Dict): Contains text feature data.
        * *indicators* (List[List[np.ndarray]]): see above.
        * *values* (List[List[np.ndarray]]): Values should be arrays of token IDs representing 
          the original strings, with zeros reserved to indicate that the text feature was not 
          recorded at a particular timestep.
        * *masks* (List[List[np.ndarray]]): A timesteps -> features nested list of attention masks for length-padded 
          token sequences.

      * *times* (List[np.ndarray]): A list of scalar arrays containing the times at which the values were recorded. Padded with zeros up to the maximum timeseries length.

      * *masks* (List[np.ndarray]): A list of arrays indicating whether each timestep is part of the episode (1) or length padding (0).

    * *event_data* (Dict): A dictionary containing event-associated data that will be used as 
      input to the Hawkes process encoder network.
      
      * *indicators* (List[List[np.ndarray]]): See above.
      * *times* (List[np.ndarray]): See above.
      * *masks* (List[np.ndarray]): See above.
    
    * *static_data* (List[np.ndarray]): A list of arrays containing static data (i.e., data that does not change over time)

    * *targets* (Dict[str, np.ndarray]): A dictionary of target arrays keyed by target names. For benchmarking with MIMIC, this should be 'mortality', 'length_of_stay', or 'phenotyping'.
    """

    def __init__(
            self,
            id: List[int],
            val_data: List[Dict[str, Dict[str, List]]],
            event_data: List[Dict[str, List[List[np.ndarray]]]],
            static_data: List[List[np.ndarray]],
            targets: List[Dict[str, np.ndarray]]
        ):

        self.patient_episodes = []
        for i in range(len(targets)):
            patient_episode = {
                'id': id[i],
                'val_data': val_data[i],
                'event_data': event_data[i],
                'static_data': static_data[i],
                'targets': targets[i]
            }
            self.patient_episodes.append(patient_episode)

    def __getitem__(self, i):
        return self.patient_episodes[i]
        
    def __len__(self):
        return len(self.patient_episodes)


class HDF5Dataset:
    """
    HDF5-backed datasets for TransEHR2 with lazy loading and optional preloading. This class is a drop-in replacement for MixedDataset that can load data lazily from HDF5 files (in RAM-constrained environments) or preload everything into RAM for maximum throughput. Use it for large datasets where the MixedDataset's pickle-based storage is inefficient.
    
    This class provides the same interface as MixedDataset but loads data from HDF5 files. It supports two modes:
    
    - preload=False (lazy): Data is loaded on-demand from disk. Good for memory-constrained environments or very large datasets.
    
    - preload=True: All data is loaded into RAM at initialization. This is much faster than pickle deserialization and eliminates per-batch I/O overhead. Recommended when you have sufficient RAM.
    
    For use with PyTorch DataLoader with num_workers > 0:
    - In lazy mode, each worker opens its own file handle.
    - In preload mode, data is shared via copy-on-write after fork.
    
    Attributes:
        h5_path: Path to the HDF5 file
        n_episodes: Number of episodes in the dataset
        preload: Whether data is preloaded into RAM
    """
    
    def __init__(self, h5_path: str, preload: bool = True):
        """
        Initialize the HDF5Dataset.
        
        Args:
            h5_path: Path to the HDF5 file created by convert_pkl_to_hdf5.py
            preload: If True, load all data into RAM at initialization (default; fast startup, zero per-batch I/O). If 
              False, load data lazily on each __getitem__ call (slow per-batch, low memory).
        """
        self.h5_path = h5_path
        self.preload = preload
        self._h5_file = None  # For lazy mode
        self._cache = None    # For preload mode
        
        # Read metadata (always fast)
        with h5py.File(h5_path, 'r') as f:
            meta = f['metadata']
            self.n_episodes = meta.attrs['n_episodes']
            self.max_ts_len_val = meta.attrs['max_ts_len_val']
            self.max_ts_len_event = meta.attrs['max_ts_len_event']
            self.n_numeric_feats = meta.attrs['n_numeric_feats']
            self.n_categorical_feats = meta.attrs['n_categorical_feats']
            self.n_text_feats = meta.attrs['n_text_feats']
            self.n_event_feats = meta.attrs['n_event_feats']
            self.static_total_dim = meta.attrs['static_total_dim']
            self.phenotype_dim = meta.attrs['phenotype_dim']
            
            # Load list-type metadata
            self.numeric_feat_dims = list(meta['numeric_feat_dims'][:]) if 'numeric_feat_dims' in meta else []
            self.categorical_feat_dims = list(meta['categorical_feat_dims'][:]) if 'categorical_feat_dims' in meta else []
            self.text_feat_dims = list(meta['text_feat_dims'][:]) if 'text_feat_dims' in meta else []
            self.static_feat_dims = list(meta['static_feat_dims'][:]) if 'static_feat_dims' in meta else []
        if preload:
            self._preload_all_data()
    
    def _preload_all_data(self):
        """Load all data from HDF5 into memory as contiguous numpy arrays."""
        print(f"Preloading {self.h5_path} into RAM...")
        
        self._cache = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            # IDs
            ids_data = f['ids'][:]
            # Handle byte strings
            if len(ids_data) > 0 and isinstance(ids_data[0], bytes):
                self._cache['ids'] = [id_.decode('utf-8') for id_ in ids_data]
            else:
                self._cache['ids'] = list(ids_data)
            
            # Value-associated data: numeric
            if self.n_numeric_feats > 0:
                self._cache['val_numeric_indicators'] = f['val_data/numeric/indicators'][:]
                self._cache['val_numeric_values'] = []
                for feat_idx in range(self.n_numeric_feats):
                    self._cache['val_numeric_values'].append(
                        f[f'val_data/numeric/values_{feat_idx}'][:]
                    )
            
            # Value-associated data: categorical
            if self.n_categorical_feats > 0:
                self._cache['val_categorical_indicators'] = f['val_data/categorical/indicators'][:]
                self._cache['val_categorical_values'] = []
                for feat_idx in range(self.n_categorical_feats):
                    self._cache['val_categorical_values'].append(
                        f[f'val_data/categorical/values_{feat_idx}'][:]
                    )
            
            # Value-associated data: text
            if self.n_text_feats > 0:
                self._cache['val_text_indicators'] = f['val_data/text/indicators'][:]
                self._cache['val_text_values'] = []
                self._cache['val_text_masks'] = []
                for feat_idx in range(self.n_text_feats):
                    self._cache['val_text_values'].append(
                        f[f'val_data/text/values_{feat_idx}'][:]
                    )
                    self._cache['val_text_masks'].append(
                        f[f'val_data/text/masks_{feat_idx}'][:]
                    )
            
            # Value-associated times and masks
            self._cache['val_times'] = f['val_data/times'][:]
            self._cache['val_masks'] = f['val_data/masks'][:]
            
            # Event-associated data
            if self.n_event_feats > 0:
                self._cache['event_indicators'] = f['event_data/indicators'][:]
            self._cache['event_times'] = f['event_data/times'][:]
            self._cache['event_masks'] = f['event_data/masks'][:]
            
            # Static data
            self._cache['static_data'] = f['static_data'][:]
            
            # Targets
            self._cache['mortality'] = f['targets/mortality'][:]
            self._cache['length_of_stay'] = f['targets/length_of_stay'][:]
            self._cache['phenotype'] = f['targets/phenotype'][:]
        
        print(f"Preload complete: {self.n_episodes} episodes loaded")
    
    @property
    def h5_file(self) -> h5py.File:
        """
        Lazy file handle initialization for multiprocessing compatibility.
        Only used when preload=False.
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5_file
    
    def __len__(self) -> int:
        return self.n_episodes
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single episode and reconstruct the nested structure.
        
        Args:
            idx: Episode index
            
        Returns:
            Dictionary with the same structure as MixedDataset.__getitem__:
            {
                'id': episode_id,
                'val_data': ValueAssociatedDataEntry,
                'event_data': EventAssociatedDataEntry,
                'static_data': StaticDataEntry,
                'targets': TargetDataEntry
            }
        """
        if self.preload:
            return self._getitem_preloaded(idx)
        else:
            return self._getitem_lazy(idx)
    
    def _getitem_preloaded(self, idx: int) -> Dict[str, Any]:
        """Get item from preloaded cache - just array slicing, very fast."""
        cache = self._cache
        
        # Episode ID
        episode_id = cache['ids'][idx]
        
        # Reconstruct val_data
        val_data = self._reconstruct_val_data_preloaded(idx)
        
        # Reconstruct event_data
        event_data = self._reconstruct_event_data_preloaded(idx)
        
        # Reconstruct static_data
        static_data = self._reconstruct_static_data_preloaded(idx)
        
        # Targets
        targets = {
            'mortality': np.array(cache['mortality'][idx], dtype=np.float32),
            'length_of_stay': np.array(cache['length_of_stay'][idx], dtype=np.float32),
            'phenotype': cache['phenotype'][idx].astype(np.float32)
        }
        
        return {
            'id': episode_id,
            'val_data': val_data,
            'event_data': event_data,
            'static_data': static_data,
            'targets': targets
        }
    
    def _reconstruct_val_data_preloaded(self, idx: int) -> Dict:
        """Reconstruct value-associated data from preloaded cache."""
        cache = self._cache
        
        val_data = {
            'numeric': {'indicators': [], 'values': []},
            'categorical': {'indicators': [], 'values': []},
            'text': {'indicators': [], 'values': [], 'masks': []},
            'times': [],
            'masks': []
        }
        
        # Numeric
        if self.n_numeric_feats > 0:
            indicators = cache['val_numeric_indicators'][idx]  # (max_ts_len, n_feats)
            for t in range(self.max_ts_len_val):
                feat_indicators = []
                feat_values = []
                for f in range(self.n_numeric_feats):
                    feat_indicators.append(np.array([indicators[t, f]], dtype=np.uint8))
                    feat_values.append(cache['val_numeric_values'][f][idx, t].astype(np.float32))
                val_data['numeric']['indicators'].append(feat_indicators)
                val_data['numeric']['values'].append(feat_values)
        else:
            for t in range(self.max_ts_len_val):
                val_data['numeric']['indicators'].append([])
                val_data['numeric']['values'].append([])
        
        # Categorical
        if self.n_categorical_feats > 0:
            indicators = cache['val_categorical_indicators'][idx]
            for t in range(self.max_ts_len_val):
                feat_indicators = []
                feat_values = []
                for f in range(self.n_categorical_feats):
                    feat_indicators.append(np.array([indicators[t, f]], dtype=np.uint8))
                    feat_values.append(cache['val_categorical_values'][f][idx, t].astype(np.int32))
                val_data['categorical']['indicators'].append(feat_indicators)
                val_data['categorical']['values'].append(feat_values)
        else:
            for t in range(self.max_ts_len_val):
                val_data['categorical']['indicators'].append([])
                val_data['categorical']['values'].append([])
        
        # Text
        if self.n_text_feats > 0:
            indicators = cache['val_text_indicators'][idx]
            for t in range(self.max_ts_len_val):
                feat_indicators = []
                feat_values = []
                feat_masks = []
                for f in range(self.n_text_feats):
                    feat_indicators.append(np.array([indicators[t, f]], dtype=np.uint8))
                    feat_values.append(cache['val_text_values'][f][idx, t].astype(np.int32))
                    feat_masks.append(cache['val_text_masks'][f][idx, t].astype(np.uint8))
                val_data['text']['indicators'].append(feat_indicators)
                val_data['text']['values'].append(feat_values)
                val_data['text']['masks'].append(feat_masks)
        else:
            for t in range(self.max_ts_len_val):
                val_data['text']['indicators'].append([])
                val_data['text']['values'].append([])
                val_data['text']['masks'].append([])
        
        # Times and masks
        times = cache['val_times'][idx]
        masks = cache['val_masks'][idx]
        for t in range(self.max_ts_len_val):
            val_data['times'].append(np.array([times[t]], dtype=np.float32))
            val_data['masks'].append(np.array([masks[t]], dtype=np.uint8))
        
        return val_data
    
    def _reconstruct_event_data_preloaded(self, idx: int) -> Dict:
        """Reconstruct event-associated data from preloaded cache."""
        cache = self._cache
        
        event_data = {
            'indicators': [],
            'times': [],
            'masks': []
        }
        
        # Indicators
        if self.n_event_feats > 0:
            indicators = cache['event_indicators'][idx]
            for t in range(self.max_ts_len_event):
                feat_indicators = []
                for f in range(self.n_event_feats):
                    feat_indicators.append(np.array([indicators[t, f]], dtype=np.uint8))
                event_data['indicators'].append(feat_indicators)
        else:
            for t in range(self.max_ts_len_event):
                event_data['indicators'].append([])
        
        # Times and masks
        times = cache['event_times'][idx]
        masks = cache['event_masks'][idx]
        for t in range(self.max_ts_len_event):
            event_data['times'].append(np.array([times[t]], dtype=np.float32))
            event_data['masks'].append(np.array([masks[t]], dtype=np.uint8))
        
        return event_data
    
    def _reconstruct_static_data_preloaded(self, idx: int) -> List[np.ndarray]:
        """Reconstruct static data from preloaded cache."""
        static_flat = self._cache['static_data'][idx]
        
        static_data = []
        offset = 0
        for feat_dim in self.static_feat_dims:
            static_data.append(static_flat[offset:offset + feat_dim].astype(np.float32))
            offset += feat_dim
        
        return static_data
    
    def _getitem_lazy(self, idx: int) -> Dict[str, Any]:
        """Get item with lazy loading from HDF5 file."""
        f = self.h5_file
        
        # Episode ID
        episode_id = f['ids'][idx]
        if isinstance(episode_id, bytes):
            episode_id = episode_id.decode('utf-8')
        
        # Reconstruct val_data
        val_data = self._reconstruct_val_data_lazy(f, idx)
        
        # Reconstruct event_data
        event_data = self._reconstruct_event_data_lazy(f, idx)
        
        # Reconstruct static_data
        static_data = self._reconstruct_static_data_lazy(f, idx)
        
        # Targets
        targets = {
            'mortality': np.array(f['targets/mortality'][idx], dtype=np.float32),
            'length_of_stay': np.array(f['targets/length_of_stay'][idx], dtype=np.float32),
            'phenotype': f['targets/phenotype'][idx].astype(np.float32)
        }
        
        return {
            'id': episode_id,
            'val_data': val_data,
            'event_data': event_data,
            'static_data': static_data,
            'targets': targets
        }
    
    def _reconstruct_val_data_lazy(self, f: h5py.File, idx: int) -> Dict:
        """Reconstruct value-associated data with lazy loading."""
        val_data = {
            'numeric': {'indicators': [], 'values': []},
            'categorical': {'indicators': [], 'values': []},
            'text': {'indicators': [], 'values': [], 'masks': []},
            'times': [],
            'masks': []
        }
        
        # Numeric
        if self.n_numeric_feats > 0:
            indicators = f['val_data/numeric/indicators'][idx]
            for t in range(self.max_ts_len_val):
                feat_indicators = []
                feat_values = []
                for feat_idx in range(self.n_numeric_feats):
                    feat_indicators.append(np.array([indicators[t, feat_idx]], dtype=np.uint8))
                    values = f[f'val_data/numeric/values_{feat_idx}'][idx, t]
                    feat_values.append(values.astype(np.float32))
                val_data['numeric']['indicators'].append(feat_indicators)
                val_data['numeric']['values'].append(feat_values)
        else:
            for t in range(self.max_ts_len_val):
                val_data['numeric']['indicators'].append([])
                val_data['numeric']['values'].append([])
        
        # Categorical
        if self.n_categorical_feats > 0:
            indicators = f['val_data/categorical/indicators'][idx]
            for t in range(self.max_ts_len_val):
                feat_indicators = []
                feat_values = []
                for feat_idx in range(self.n_categorical_feats):
                    feat_indicators.append(np.array([indicators[t, feat_idx]], dtype=np.uint8))
                    values = f[f'val_data/categorical/values_{feat_idx}'][idx, t]
                    feat_values.append(values.astype(np.int32))
                val_data['categorical']['indicators'].append(feat_indicators)
                val_data['categorical']['values'].append(feat_values)
        else:
            for t in range(self.max_ts_len_val):
                val_data['categorical']['indicators'].append([])
                val_data['categorical']['values'].append([])
        
        # Text
        if self.n_text_feats > 0:
            indicators = f['val_data/text/indicators'][idx]
            for t in range(self.max_ts_len_val):
                feat_indicators = []
                feat_values = []
                feat_masks = []
                for feat_idx in range(self.n_text_feats):
                    feat_indicators.append(np.array([indicators[t, feat_idx]], dtype=np.uint8))
                    values = f[f'val_data/text/values_{feat_idx}'][idx, t]
                    feat_values.append(values.astype(np.int32))
                    masks = f[f'val_data/text/masks_{feat_idx}'][idx, t]
                    feat_masks.append(masks.astype(np.uint8))
                val_data['text']['indicators'].append(feat_indicators)
                val_data['text']['values'].append(feat_values)
                val_data['text']['masks'].append(feat_masks)
        else:
            for t in range(self.max_ts_len_val):
                val_data['text']['indicators'].append([])
                val_data['text']['values'].append([])
                val_data['text']['masks'].append([])
        
        # Times and masks
        times = f['val_data/times'][idx]
        masks = f['val_data/masks'][idx]
        for t in range(self.max_ts_len_val):
            val_data['times'].append(np.array([times[t]], dtype=np.float32))
            val_data['masks'].append(np.array([masks[t]], dtype=np.uint8))
        
        return val_data
    
    def _reconstruct_event_data_lazy(self, f: h5py.File, idx: int) -> Dict:
        """Reconstruct event-associated data with lazy loading."""
        event_data = {
            'indicators': [],
            'times': [],
            'masks': []
        }
        
        if self.n_event_feats > 0:
            indicators = f['event_data/indicators'][idx]
            for t in range(self.max_ts_len_event):
                feat_indicators = []
                for feat_idx in range(self.n_event_feats):
                    feat_indicators.append(np.array([indicators[t, feat_idx]], dtype=np.uint8))
                event_data['indicators'].append(feat_indicators)
        else:
            for t in range(self.max_ts_len_event):
                event_data['indicators'].append([])
        
        times = f['event_data/times'][idx]
        masks = f['event_data/masks'][idx]
        for t in range(self.max_ts_len_event):
            event_data['times'].append(np.array([times[t]], dtype=np.float32))
            event_data['masks'].append(np.array([masks[t]], dtype=np.uint8))
        
        return event_data
    
    def _reconstruct_static_data_lazy(self, f: h5py.File, idx: int) -> List[np.ndarray]:
        """Reconstruct static data with lazy loading."""
        static_flat = f['static_data'][idx]
        
        static_data = []
        offset = 0
        for feat_dim in self.static_feat_dims:
            static_data.append(static_flat[offset:offset + feat_dim].astype(np.float32))
            offset += feat_dim
        
        return static_data
    
    def __del__(self):
        """Clean up file handle on deletion."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass
    
    def close(self):
        """Explicitly close the HDF5 file handle and free cache."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._cache = None
        