"""
Convert TransEHR2 pickle files to HDF5 format for faster loading.

This script converts the nested list-of-lists structure in pickle files to
contiguous numpy arrays stored in HDF5, enabling memory-mapped lazy loading.

Usage:
    python convert_pkl_to_hdf5.py --data_dir /path/to/pkl/files --output_dir /path/to/hdf5/files

The script will convert train.pkl, val.pkl (if exists), and test.pkl to corresponding .h5 files.
"""

import argparse
import h5py
import numpy as np
import os
import pickle
import sys

from tqdm import tqdm
from typing import Dict, List


def flatten_nested_list_to_array(
    nested_list: List[List[np.ndarray]], 
    n_timesteps: int,
    n_features: int,
    feature_dim: int,
    dtype: np.dtype
) -> np.ndarray:
    """
    Convert List[List[np.ndarray]] structure to a contiguous 3D or 4D array.
    
    Args:
        nested_list: [timestep][feature] -> np.ndarray of shape (feature_dim,)
        n_timesteps: Number of timesteps (outer list length)
        n_features: Number of features (inner list length)
        feature_dim: Dimension of each feature array
        dtype: Target numpy dtype
        
    Returns:
        np.ndarray of shape (n_timesteps, n_features, feature_dim)
    """
    # Pre-allocate output array
    out = np.zeros((n_timesteps, n_features, feature_dim), dtype=dtype)
    
    for t in range(n_timesteps):
        for f in range(n_features):
            arr = nested_list[t][f]
            out[t, f, :len(arr)] = arr
    
    return out


def flatten_timestep_list_to_array(
    timestep_list: List[np.ndarray],
    n_timesteps: int,
    dtype: np.dtype
) -> np.ndarray:
    """
    Convert List[np.ndarray] (each shape (1,)) to a 1D array.
    
    Args:
        timestep_list: [timestep] -> np.ndarray of shape (1,)
        n_timesteps: Number of timesteps
        dtype: Target numpy dtype
        
    Returns:
        np.ndarray of shape (n_timesteps,)
    """
    out = np.zeros(n_timesteps, dtype=dtype)
    for t in range(n_timesteps):
        out[t] = timestep_list[t][0]
    return out


def flatten_static_data(static_data: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate list of feature arrays into a single 1D array.
    
    Args:
        static_data: List of np.ndarray, each of shape (feature_dim,)
        
    Returns:
        np.ndarray of shape (total_dim,)
    """
    return np.concatenate(static_data, axis=0)


def get_feature_dims(val_data_list: List[Dict], event_data_list: List[Dict], static_data_list: List[List]) -> Dict:
    """
    Infer feature dimensions from the first episode's data.
    
    Returns a dictionary with dimension info for array pre-allocation.
    """
    # Use first episode to infer dimensions
    val_data = val_data_list[0]
    event_data = event_data_list[0]
    static_data = static_data_list[0]
    
    # Get max timeseries length (should be consistent due to padding)
    max_ts_len_val = len(val_data['times'])
    max_ts_len_event = len(event_data['times'])
    
    # Get number of features for each type
    n_numeric_feats = len(val_data['numeric']['indicators'][0]) if val_data['numeric']['indicators'] else 0
    n_categorical_feats = len(val_data['categorical']['indicators'][0]) if val_data['categorical']['indicators'] else 0
    n_text_feats = len(val_data['text']['indicators'][0]) if val_data['text']['indicators'] else 0
    n_event_feats = len(event_data['indicators'][0]) if event_data['indicators'] else 0
    
    # Get feature dimensions (size of each feature's value array)
    numeric_feat_dims = []
    if n_numeric_feats > 0:
        for f in range(n_numeric_feats):
            numeric_feat_dims.append(len(val_data['numeric']['values'][0][f]))
    
    categorical_feat_dims = []
    if n_categorical_feats > 0:
        for f in range(n_categorical_feats):
            categorical_feat_dims.append(len(val_data['categorical']['values'][0][f]))
    
    text_feat_dims = []
    if n_text_feats > 0:
        for f in range(n_text_feats):
            text_feat_dims.append(len(val_data['text']['values'][0][f]))
    
    # Static data: list of arrays, compute total dimension
    static_total_dim = sum(len(arr) for arr in static_data)
    static_feat_dims = [len(arr) for arr in static_data]
    
    # Target dimensions
    targets = None
    
    return {
        'max_ts_len_val': max_ts_len_val,
        'max_ts_len_event': max_ts_len_event,
        'n_numeric_feats': n_numeric_feats,
        'n_categorical_feats': n_categorical_feats,
        'n_text_feats': n_text_feats,
        'n_event_feats': n_event_feats,
        'numeric_feat_dims': numeric_feat_dims,
        'categorical_feat_dims': categorical_feat_dims,
        'text_feat_dims': text_feat_dims,
        'static_total_dim': static_total_dim,
        'static_feat_dims': static_feat_dims,
    }


def convert_partition(pkl_path: str, h5_path: str, compression: str = 'gzip', compression_opts: int = 4):
    """
    Convert a single partition pickle file to HDF5 format.
    
    Args:
        pkl_path: Path to input pickle file
        h5_path: Path to output HDF5 file
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_opts: Compression level (1-9 for gzip)
    """
    print(f"Loading {pkl_path}...")
    sys.stdout.flush()
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    ids = data['id']
    val_data_list = data['val_data']
    event_data_list = data['event_data']
    static_data_list = data['static_data']
    targets_list = data['targets']
    
    n_episodes = len(ids)
    print(f"Converting {n_episodes} episodes...")
    sys.stdout.flush()
    
    # Infer dimensions from data
    dims = get_feature_dims(val_data_list, event_data_list, static_data_list)
    
    # Create HDF5 file
    with h5py.File(h5_path, 'w') as h5f:
        # Store metadata
        meta = h5f.create_group('metadata')
        meta.attrs['n_episodes'] = n_episodes
        for key, value in dims.items():
            if isinstance(value, list):
                meta.create_dataset(key, data=np.array(value, dtype=np.int32))
            else:
                meta.attrs[key] = value
        
        # Store IDs
        # Handle both string and integer IDs
        if isinstance(ids[0], str):
            dt = h5py.special_dtype(vlen=str)
            h5f.create_dataset('ids', data=ids, dtype=dt)
        else:
            h5f.create_dataset('ids', data=np.array(ids))
        
        # Create groups for data types
        val_grp = h5f.create_group('val_data')
        event_grp = h5f.create_group('event_data')
        
        # Compression settings
        comp_kwargs = {}
        if compression:
            comp_kwargs = {'compression': compression, 'compression_opts': compression_opts}
        
        # Pre-allocate datasets for value-associated data
        # Numeric
        if dims['n_numeric_feats'] > 0:
            numeric_grp = val_grp.create_group('numeric')
            # Indicators: (n_episodes, max_ts_len, n_features)
            numeric_grp.create_dataset(
                'indicators',
                shape=(n_episodes, dims['max_ts_len_val'], dims['n_numeric_feats']),
                dtype=np.uint8,
                **comp_kwargs
            )
            # Values: stored per-feature since dimensions may vary
            # Shape: (n_episodes, max_ts_len, feat_dim) for each feature
            for f, feat_dim in enumerate(dims['numeric_feat_dims']):
                numeric_grp.create_dataset(
                    f'values_{f}',
                    shape=(n_episodes, dims['max_ts_len_val'], feat_dim),
                    dtype=np.float32,
                    **comp_kwargs
                )
        
        # Categorical
        if dims['n_categorical_feats'] > 0:
            categorical_grp = val_grp.create_group('categorical')
            categorical_grp.create_dataset(
                'indicators',
                shape=(n_episodes, dims['max_ts_len_val'], dims['n_categorical_feats']),
                dtype=np.uint8,
                **comp_kwargs
            )
            for f, feat_dim in enumerate(dims['categorical_feat_dims']):
                categorical_grp.create_dataset(
                    f'values_{f}',
                    shape=(n_episodes, dims['max_ts_len_val'], feat_dim),
                    dtype=np.int32,
                    **comp_kwargs
                )
        
        # Text
        if dims['n_text_feats'] > 0:
            text_grp = val_grp.create_group('text')
            text_grp.create_dataset(
                'indicators',
                shape=(n_episodes, dims['max_ts_len_val'], dims['n_text_feats']),
                dtype=np.uint8,
                **comp_kwargs
            )
            for f, feat_dim in enumerate(dims['text_feat_dims']):
                text_grp.create_dataset(
                    f'values_{f}',
                    shape=(n_episodes, dims['max_ts_len_val'], feat_dim),
                    dtype=np.int32,
                    **comp_kwargs
                )
                text_grp.create_dataset(
                    f'masks_{f}',
                    shape=(n_episodes, dims['max_ts_len_val'], feat_dim),
                    dtype=np.uint8,
                    **comp_kwargs
                )
        
        # Times and masks for val_data
        val_grp.create_dataset(
            'times',
            shape=(n_episodes, dims['max_ts_len_val']),
            dtype=np.float32,
            **comp_kwargs
        )
        val_grp.create_dataset(
            'masks',
            shape=(n_episodes, dims['max_ts_len_val']),
            dtype=np.uint8,
            **comp_kwargs
        )
        
        # Event data
        if dims['n_event_feats'] > 0:
            event_grp.create_dataset(
                'indicators',
                shape=(n_episodes, dims['max_ts_len_event'], dims['n_event_feats']),
                dtype=np.uint8,
                **comp_kwargs
            )
        event_grp.create_dataset(
            'times',
            shape=(n_episodes, dims['max_ts_len_event']),
            dtype=np.float32,
            **comp_kwargs
        )
        event_grp.create_dataset(
            'masks',
            shape=(n_episodes, dims['max_ts_len_event']),
            dtype=np.uint8,
            **comp_kwargs
        )
        
        # Static data: (n_episodes, total_static_dim)
        h5f.create_dataset(
            'static_data',
            shape=(n_episodes, dims['static_total_dim']),
            dtype=np.float32,
            **comp_kwargs
        )
        
        # Targets
        # Infer phenotype dimension from first episode
        phenotype_dim = len(targets_list[0]['phenotype'])
        targets_grp = h5f.create_group('targets')
        targets_grp.create_dataset('mortality', shape=(n_episodes,), dtype=np.float32, **comp_kwargs)
        targets_grp.create_dataset('length_of_stay', shape=(n_episodes,), dtype=np.float32, **comp_kwargs)
        targets_grp.create_dataset('phenotype', shape=(n_episodes, phenotype_dim), dtype=np.float32, **comp_kwargs)
        
        # Store phenotype dimension in metadata
        meta.attrs['phenotype_dim'] = phenotype_dim
        
        # Now fill the datasets
        for i in tqdm(range(n_episodes), desc="Converting episodes"):
            val_data = val_data_list[i]
            event_data = event_data_list[i]
            static_data = static_data_list[i]
            targets = targets_list[i]
            
            # Value-associated data: numeric
            if dims['n_numeric_feats'] > 0:
                # Indicators
                for t in range(dims['max_ts_len_val']):
                    for f in range(dims['n_numeric_feats']):
                        h5f['val_data/numeric/indicators'][i, t, f] = val_data['numeric']['indicators'][t][f][0]
                # Values (per-feature)
                for f, feat_dim in enumerate(dims['numeric_feat_dims']):
                    for t in range(dims['max_ts_len_val']):
                        arr = val_data['numeric']['values'][t][f]
                        h5f[f'val_data/numeric/values_{f}'][i, t, :len(arr)] = arr
            
            # Value-associated data: categorical
            if dims['n_categorical_feats'] > 0:
                for t in range(dims['max_ts_len_val']):
                    for f in range(dims['n_categorical_feats']):
                        h5f['val_data/categorical/indicators'][i, t, f] = val_data['categorical']['indicators'][t][f][0]
                for f, feat_dim in enumerate(dims['categorical_feat_dims']):
                    for t in range(dims['max_ts_len_val']):
                        arr = val_data['categorical']['values'][t][f]
                        h5f[f'val_data/categorical/values_{f}'][i, t, :len(arr)] = arr
            
            # Value-associated data: text
            if dims['n_text_feats'] > 0:
                for t in range(dims['max_ts_len_val']):
                    for f in range(dims['n_text_feats']):
                        h5f['val_data/text/indicators'][i, t, f] = val_data['text']['indicators'][t][f][0]
                for f, feat_dim in enumerate(dims['text_feat_dims']):
                    for t in range(dims['max_ts_len_val']):
                        arr = val_data['text']['values'][t][f]
                        h5f[f'val_data/text/values_{f}'][i, t, :len(arr)] = arr
                        mask = val_data['text']['masks'][t][f]
                        h5f[f'val_data/text/masks_{f}'][i, t, :len(mask)] = mask
            
            # Times and masks for val_data
            for t in range(dims['max_ts_len_val']):
                h5f['val_data/times'][i, t] = val_data['times'][t][0]
                h5f['val_data/masks'][i, t] = val_data['masks'][t][0]
            
            # Event data
            if dims['n_event_feats'] > 0:
                for t in range(dims['max_ts_len_event']):
                    for f in range(dims['n_event_feats']):
                        h5f['event_data/indicators'][i, t, f] = event_data['indicators'][t][f][0]
            for t in range(dims['max_ts_len_event']):
                h5f['event_data/times'][i, t] = event_data['times'][t][0]
                h5f['event_data/masks'][i, t] = event_data['masks'][t][0]
            
            # Static data
            static_flat = flatten_static_data(static_data)
            h5f['static_data'][i, :] = static_flat
            
            # Targets
            h5f['targets/mortality'][i] = targets['mortality']
            h5f['targets/length_of_stay'][i] = targets['length_of_stay']
            h5f['targets/phenotype'][i, :] = targets['phenotype']
        
        print(f"Saved {h5_path}")
        
    # Free memory
    del data, val_data_list, event_data_list, static_data_list, targets_list


def main():
    parser = argparse.ArgumentParser(description='Convert TransEHR2 pickle files to HDF5 format')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pkl files')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for .h5 files (default: same as data_dir)')
    parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf', 'none'],
                        help='Compression algorithm (default: gzip)')
    parser.add_argument('--compression_level', type=int, default=4,
                        help='Compression level for gzip (1-9, default: 4)')
    parser.add_argument('--partitions', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Partitions to convert (default: train val test)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    compression = None if args.compression == 'none' else args.compression
    
    for partition in args.partitions:
        pkl_path = os.path.join(args.data_dir, f'{partition}.pkl')
        h5_path = os.path.join(args.output_dir, f'{partition}.h5')
        
        if not os.path.exists(pkl_path):
            if partition == 'val':
                print(f"Skipping {partition} (file not found)")
                continue
            else:
                raise FileNotFoundError(f"{pkl_path} not found")
        
        convert_partition(pkl_path, h5_path, compression=compression, compression_opts=args.compression_level)
    
    # Copy summary statistics if present
    stats_path = os.path.join(args.data_dir, 'summary_statistics_train.npz')
    if os.path.exists(stats_path) and args.output_dir != args.data_dir:
        import shutil
        shutil.copy(stats_path, args.output_dir)
        print(f"Copied summary_statistics_train.npz to {args.output_dir}")
    
    print("Conversion complete!")


if __name__ == '__main__':
    main()
