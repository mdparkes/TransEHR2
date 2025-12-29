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
    A dataset backed by HDF5 files with hybrid sparse storage.
    
    Numeric, categorical, and event features are stored without right-padding.
    Text features are stored sparsely (only timesteps with actual text).
    On read, data is reconstructed to the dense padded format expected by
    the collation and model code.
    
    Supports two modes:
    - preload=True: Load all data into RAM at init (recommended with sufficient RAM)
    - preload=False: Load data lazily per __getitem__ call
    """
    
    def __init__(self, h5_path: str, preload: bool = True):
        """
        Initialize the HDF5Dataset.
        
        Args:
            h5_path: Path to HDF5 file created by extract_mimic_hdf5
            preload: If True, load all data into RAM at initialization
        """
        
        self.h5_path = h5_path
        self.preload = preload
        self._h5_file = None
        self._cache = None
        
        # Read metadata
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
            
            self.numeric_feat_dims = list(meta['numeric_feat_dims'][:]) if 'numeric_feat_dims' in meta else []
            self.categorical_feat_dims = list(meta['categorical_feat_dims'][:]) if 'categorical_feat_dims' in meta else []
            self.text_feat_dims = list(meta['text_feat_dims'][:]) if 'text_feat_dims' in meta else []
            self.static_feat_dims = list(meta['static_feat_dims'][:]) if 'static_feat_dims' in meta else []
        
        if preload:
            self._preload_all_data()
    
    def _preload_all_data(self):
        """Load all data from HDF5 into memory."""
        
        print(f"Preloading {self.h5_path} into RAM...")
        
        self._cache = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            # IDs
            ids_data = f['ids'][:]
            if len(ids_data) > 0 and isinstance(ids_data[0], bytes):
                self._cache['ids'] = [id_.decode('utf-8') for id_ in ids_data]
            else:
                self._cache['ids'] = list(ids_data)
            
            # Val data
            self._cache['val_episode_offsets'] = f['val_data/episode_offsets'][:]
            self._cache['val_times'] = f['val_data/times'][:]
            
            if self.n_numeric_feats > 0:
                self._cache['val_numeric_indicators'] = f['val_data/numeric/indicators'][:]
                self._cache['val_numeric_values'] = []
                for feat_idx in range(self.n_numeric_feats):
                    self._cache['val_numeric_values'].append(f[f'val_data/numeric/values_{feat_idx}'][:])
            
            if self.n_categorical_feats > 0:
                self._cache['val_categorical_indicators'] = f['val_data/categorical/indicators'][:]
                self._cache['val_categorical_values'] = []
                for feat_idx in range(self.n_categorical_feats):
                    self._cache['val_categorical_values'].append(f[f'val_data/categorical/values_{feat_idx}'][:])
            
            if self.n_text_feats > 0:
                self._cache['val_text_indicators'] = f['val_data/text/indicators'][:]
                self._cache['val_text_feats'] = []
                for feat_idx in range(self.n_text_feats):
                    feat_data = {
                        'episode_offsets': f[f'val_data/text/feat_{feat_idx}/episode_offsets'][:],
                        'timestep_indices': f[f'val_data/text/feat_{feat_idx}/timestep_indices'][:],
                        'values': f[f'val_data/text/feat_{feat_idx}/values'][:],
                        'masks': f[f'val_data/text/feat_{feat_idx}/masks'][:]
                    }
                    self._cache['val_text_feats'].append(feat_data)
            
            # Event data
            self._cache['event_episode_offsets'] = f['event_data/episode_offsets'][:]
            self._cache['event_times'] = f['event_data/times'][:]
            if self.n_event_feats > 0:
                self._cache['event_indicators'] = f['event_data/indicators'][:]
            
            # Static data
            self._cache['static_data'] = f['static_data'][:]
            
            # Targets
            self._cache['mortality'] = f['targets/mortality'][:]
            self._cache['length_of_stay'] = f['targets/length_of_stay'][:]
            self._cache['phenotype'] = f['targets/phenotype'][:]
        
        print(f"Preload complete: {self.n_episodes} episodes loaded")
    
    @property
    def h5_file(self):
        """Lazy file handle for non-preloaded access."""

        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5_file
    
    def __len__(self) -> int:
        return self.n_episodes
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and reconstruct a single episode to dense padded format.
        
        Returns dictionary matching MixedDataset.__getitem__ structure.
        """
        if self.preload:
            return self._getitem_preloaded(idx)
        else:
            return self._getitem_lazy(idx)
    
    def _getitem_preloaded(self, idx: int) -> Dict[str, Any]:
        """Get item from preloaded cache."""
        cache = self._cache
        
        episode_id = cache['ids'][idx]
        
        # Get val_data slice boundaries
        val_start = cache['val_episode_offsets'][idx]
        val_end = cache['val_episode_offsets'][idx + 1]
        val_len = val_end - val_start
        
        # Reconstruct val_data
        val_data = self._reconstruct_val_data_preloaded(idx, val_start, val_end, val_len)
        
        # Get event_data slice boundaries
        event_start = cache['event_episode_offsets'][idx]
        event_end = cache['event_episode_offsets'][idx + 1]
        event_len = event_end - event_start
        
        # Reconstruct event_data
        event_data = self._reconstruct_event_data_preloaded(event_start, event_end, event_len)
        
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
    
    def _reconstruct_val_data_preloaded(self, idx: int, val_start: int, val_end: int, val_len: int) -> Dict:
        """Reconstruct dense padded val_data from sparse storage."""
        cache = self._cache
        max_ts = self.max_ts_len_val
        
        val_data = {
            'numeric': {'indicators': [], 'values': []},
            'categorical': {'indicators': [], 'values': []},
            'text': {'indicators': [], 'values': [], 'masks': []},
            'times': [],
            'masks': []
        }
        
        # Get times for real timesteps
        real_times = cache['val_times'][val_start:val_end]
        
        # Reconstruct numeric
        if self.n_numeric_feats > 0:
            real_indicators = cache['val_numeric_indicators'][val_start:val_end]
            real_values = [cache['val_numeric_values'][f][val_start:val_end] for f in range(self.n_numeric_feats)]
            
            for t in range(max_ts):
                if t < val_len:
                    feat_indicators = [np.array([real_indicators[t, f]], dtype=np.uint8) 
                                       for f in range(self.n_numeric_feats)]
                    feat_values = [real_values[f][t].astype(np.float32) 
                                   for f in range(self.n_numeric_feats)]
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_numeric_feats)]
                    feat_values = [np.zeros(self.numeric_feat_dims[f], dtype=np.float32) 
                                   for f in range(self.n_numeric_feats)]
                val_data['numeric']['indicators'].append(feat_indicators)
                val_data['numeric']['values'].append(feat_values)
        else:
            for t in range(max_ts):
                val_data['numeric']['indicators'].append([])
                val_data['numeric']['values'].append([])
        
        # Reconstruct categorical
        if self.n_categorical_feats > 0:
            real_indicators = cache['val_categorical_indicators'][val_start:val_end]
            real_values = [cache['val_categorical_values'][f][val_start:val_end] for f in range(self.n_categorical_feats)]
            
            for t in range(max_ts):
                if t < val_len:
                    feat_indicators = [np.array([real_indicators[t, f]], dtype=np.uint8) 
                                       for f in range(self.n_categorical_feats)]
                    feat_values = [real_values[f][t].astype(np.int32) 
                                   for f in range(self.n_categorical_feats)]
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_categorical_feats)]
                    feat_values = [np.zeros(self.categorical_feat_dims[f], dtype=np.int32) 
                                   for f in range(self.n_categorical_feats)]
                val_data['categorical']['indicators'].append(feat_indicators)
                val_data['categorical']['values'].append(feat_values)
        else:
            for t in range(max_ts):
                val_data['categorical']['indicators'].append([])
                val_data['categorical']['values'].append([])
        
        # Reconstruct text (sparse within timesteps)
        if self.n_text_feats > 0:
            real_indicators = cache['val_text_indicators'][val_start:val_end]
            
            # Build per-feature maps of timestep -> text data
            text_maps = []
            for f in range(self.n_text_feats):
                feat_data = cache['val_text_feats'][f]
                text_start = feat_data['episode_offsets'][idx]
                text_end = feat_data['episode_offsets'][idx + 1]
                
                timestep_indices = feat_data['timestep_indices'][text_start:text_end]
                values = feat_data['values'][text_start:text_end]
                masks = feat_data['masks'][text_start:text_end]
                
                # Map timestep index -> (values, masks)
                ts_map = {}
                for j, ts_idx in enumerate(timestep_indices):
                    ts_map[ts_idx] = (values[j], masks[j])
                text_maps.append(ts_map)
            
            for t in range(max_ts):
                if t < val_len:
                    feat_indicators = [np.array([real_indicators[t, f]], dtype=np.uint8) 
                                       for f in range(self.n_text_feats)]
                    feat_values = []
                    feat_masks = []
                    for f in range(self.n_text_feats):
                        if t in text_maps[f]:
                            feat_values.append(text_maps[f][t][0].astype(np.int32))
                            feat_masks.append(text_maps[f][t][1].astype(np.uint8))
                        else:
                            feat_values.append(np.zeros(self.text_feat_dims[f], dtype=np.int32))
                            feat_masks.append(np.zeros(self.text_feat_dims[f], dtype=np.uint8))
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_text_feats)]
                    feat_values = [np.zeros(self.text_feat_dims[f], dtype=np.int32) 
                                   for f in range(self.n_text_feats)]
                    feat_masks = [np.zeros(self.text_feat_dims[f], dtype=np.uint8) 
                                  for f in range(self.n_text_feats)]
                val_data['text']['indicators'].append(feat_indicators)
                val_data['text']['values'].append(feat_values)
                val_data['text']['masks'].append(feat_masks)
        else:
            for t in range(max_ts):
                val_data['text']['indicators'].append([])
                val_data['text']['values'].append([])
                val_data['text']['masks'].append([])
        
        # Reconstruct times and masks
        for t in range(max_ts):
            if t < val_len:
                val_data['times'].append(np.array([real_times[t]], dtype=np.float32))
                val_data['masks'].append(np.array([1], dtype=np.uint8))
            else:
                val_data['times'].append(np.array([0.0], dtype=np.float32))
                val_data['masks'].append(np.array([0], dtype=np.uint8))
        
        return val_data
    
    def _reconstruct_event_data_preloaded(self, event_start: int, event_end: int, event_len: int) -> Dict:
        """Reconstruct dense padded event_data from sparse storage."""
        cache = self._cache
        max_ts = self.max_ts_len_event
        
        event_data = {
            'indicators': [],
            'times': [],
            'masks': []
        }
        
        real_times = cache['event_times'][event_start:event_end]
        
        if self.n_event_feats > 0:
            real_indicators = cache['event_indicators'][event_start:event_end]
            
            for t in range(max_ts):
                if t < event_len:
                    feat_indicators = [np.array([real_indicators[t, f]], dtype=np.uint8) 
                                       for f in range(self.n_event_feats)]
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_event_feats)]
                event_data['indicators'].append(feat_indicators)
        else:
            for t in range(max_ts):
                event_data['indicators'].append([])
        
        for t in range(max_ts):
            if t < event_len:
                event_data['times'].append(np.array([real_times[t]], dtype=np.float32))
                event_data['masks'].append(np.array([1], dtype=np.uint8))
            else:
                event_data['times'].append(np.array([0.0], dtype=np.float32))
                event_data['masks'].append(np.array([0], dtype=np.uint8))
        
        return event_data
    
    def _reconstruct_static_data_preloaded(self, idx: int) -> List[np.ndarray]:
        """Reconstruct static_data list from flat array."""
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
        
        # Get episode ID
        episode_id = f['ids'][idx]
        if isinstance(episode_id, bytes):
            episode_id = episode_id.decode('utf-8')
        
        # Get val_data slice boundaries
        val_offsets = f['val_data/episode_offsets'][:]
        val_start = val_offsets[idx]
        val_end = val_offsets[idx + 1]
        val_len = val_end - val_start
        
        # Reconstruct val_data
        val_data = self._reconstruct_val_data_lazy(f, idx, val_start, val_end, val_len)
        
        # Get event_data slice boundaries
        event_offsets = f['event_data/episode_offsets'][:]
        event_start = event_offsets[idx]
        event_end = event_offsets[idx + 1]
        event_len = event_end - event_start
        
        # Reconstruct event_data
        event_data = self._reconstruct_event_data_lazy(f, event_start, event_end, event_len)
        
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
    
    def _reconstruct_val_data_lazy(self, f, idx: int, val_start: int, val_end: int, val_len: int) -> Dict:
        """Reconstruct dense padded val_data with lazy loading."""
        max_ts = self.max_ts_len_val
        
        val_data = {
            'numeric': {'indicators': [], 'values': []},
            'categorical': {'indicators': [], 'values': []},
            'text': {'indicators': [], 'values': [], 'masks': []},
            'times': [],
            'masks': []
        }
        
        real_times = f['val_data/times'][val_start:val_end]
        
        # Numeric
        if self.n_numeric_feats > 0:
            real_indicators = f['val_data/numeric/indicators'][val_start:val_end]
            real_values = [f[f'val_data/numeric/values_{feat}'][val_start:val_end] 
                           for feat in range(self.n_numeric_feats)]
            
            for t in range(max_ts):
                if t < val_len:
                    feat_indicators = [np.array([real_indicators[t, feat]], dtype=np.uint8) 
                                       for feat in range(self.n_numeric_feats)]
                    feat_values = [real_values[feat][t].astype(np.float32) 
                                   for feat in range(self.n_numeric_feats)]
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_numeric_feats)]
                    feat_values = [np.zeros(self.numeric_feat_dims[feat], dtype=np.float32) 
                                   for feat in range(self.n_numeric_feats)]
                val_data['numeric']['indicators'].append(feat_indicators)
                val_data['numeric']['values'].append(feat_values)
        else:
            for t in range(max_ts):
                val_data['numeric']['indicators'].append([])
                val_data['numeric']['values'].append([])
        
        # Categorical
        if self.n_categorical_feats > 0:
            real_indicators = f['val_data/categorical/indicators'][val_start:val_end]
            real_values = [f[f'val_data/categorical/values_{feat}'][val_start:val_end] 
                           for feat in range(self.n_categorical_feats)]
            
            for t in range(max_ts):
                if t < val_len:
                    feat_indicators = [np.array([real_indicators[t, feat]], dtype=np.uint8) 
                                       for feat in range(self.n_categorical_feats)]
                    feat_values = [real_values[feat][t].astype(np.int32) 
                                   for feat in range(self.n_categorical_feats)]
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_categorical_feats)]
                    feat_values = [np.zeros(self.categorical_feat_dims[feat], dtype=np.int32) 
                                   for feat in range(self.n_categorical_feats)]
                val_data['categorical']['indicators'].append(feat_indicators)
                val_data['categorical']['values'].append(feat_values)
        else:
            for t in range(max_ts):
                val_data['categorical']['indicators'].append([])
                val_data['categorical']['values'].append([])
        
        # Text (sparse)
        if self.n_text_feats > 0:
            real_indicators = f['val_data/text/indicators'][val_start:val_end]
            
            text_maps = []
            for feat in range(self.n_text_feats):
                text_offsets = f[f'val_data/text/feat_{feat}/episode_offsets'][:]
                text_start = text_offsets[idx]
                text_end = text_offsets[idx + 1]
                
                timestep_indices = f[f'val_data/text/feat_{feat}/timestep_indices'][text_start:text_end]
                values = f[f'val_data/text/feat_{feat}/values'][text_start:text_end]
                masks = f[f'val_data/text/feat_{feat}/masks'][text_start:text_end]
                
                ts_map = {}
                for j, ts_idx in enumerate(timestep_indices):
                    ts_map[ts_idx] = (values[j], masks[j])
                text_maps.append(ts_map)
            
            for t in range(max_ts):
                if t < val_len:
                    feat_indicators = [np.array([real_indicators[t, feat]], dtype=np.uint8) 
                                       for feat in range(self.n_text_feats)]
                    feat_values = []
                    feat_masks = []
                    for feat in range(self.n_text_feats):
                        if t in text_maps[feat]:
                            feat_values.append(text_maps[feat][t][0].astype(np.int32))
                            feat_masks.append(text_maps[feat][t][1].astype(np.uint8))
                        else:
                            feat_values.append(np.zeros(self.text_feat_dims[feat], dtype=np.int32))
                            feat_masks.append(np.zeros(self.text_feat_dims[feat], dtype=np.uint8))
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_text_feats)]
                    feat_values = [np.zeros(self.text_feat_dims[feat], dtype=np.int32) 
                                   for feat in range(self.n_text_feats)]
                    feat_masks = [np.zeros(self.text_feat_dims[feat], dtype=np.uint8) 
                                  for feat in range(self.n_text_feats)]
                val_data['text']['indicators'].append(feat_indicators)
                val_data['text']['values'].append(feat_values)
                val_data['text']['masks'].append(feat_masks)
        else:
            for t in range(max_ts):
                val_data['text']['indicators'].append([])
                val_data['text']['values'].append([])
                val_data['text']['masks'].append([])
        
        # Times and masks
        for t in range(max_ts):
            if t < val_len:
                val_data['times'].append(np.array([real_times[t]], dtype=np.float32))
                val_data['masks'].append(np.array([1], dtype=np.uint8))
            else:
                val_data['times'].append(np.array([0.0], dtype=np.float32))
                val_data['masks'].append(np.array([0], dtype=np.uint8))
        
        return val_data
    
    def _reconstruct_event_data_lazy(self, f, event_start: int, event_end: int, event_len: int) -> Dict:
        """Reconstruct dense padded event_data with lazy loading."""
        max_ts = self.max_ts_len_event
        
        event_data = {
            'indicators': [],
            'times': [],
            'masks': []
        }
        
        real_times = f['event_data/times'][event_start:event_end]
        
        if self.n_event_feats > 0:
            real_indicators = f['event_data/indicators'][event_start:event_end]
            
            for t in range(max_ts):
                if t < event_len:
                    feat_indicators = [np.array([real_indicators[t, feat]], dtype=np.uint8) 
                                       for feat in range(self.n_event_feats)]
                else:
                    feat_indicators = [np.array([0], dtype=np.uint8) for _ in range(self.n_event_feats)]
                event_data['indicators'].append(feat_indicators)
        else:
            for t in range(max_ts):
                event_data['indicators'].append([])
        
        for t in range(max_ts):
            if t < event_len:
                event_data['times'].append(np.array([real_times[t]], dtype=np.float32))
                event_data['masks'].append(np.array([1], dtype=np.uint8))
            else:
                event_data['times'].append(np.array([0.0], dtype=np.float32))
                event_data['masks'].append(np.array([0], dtype=np.uint8))
        
        return event_data
    
    def _reconstruct_static_data_lazy(self, f, idx: int) -> List[np.ndarray]:
        """Reconstruct static_data list with lazy loading."""
        static_flat = f['static_data'][idx]
        
        static_data = []
        offset = 0
        for feat_dim in self.static_feat_dims:
            static_data.append(static_flat[offset:offset + feat_dim].astype(np.float32))
            offset += feat_dim
        
        return static_data
    
    def __del__(self):
        """Clean up file handle."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass
    
    def close(self):
        """Explicitly close file handle and free cache."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._cache = None
