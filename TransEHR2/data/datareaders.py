import os
import pandas as pd
import re

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


class MIMICDataReader(Sequence):
    """Base reader class for MIMIC datasets.
    
    Readers that extend this class will be able to read data from specific versions of the MIMIC dataset.
    """

    def __init__(
        self,
        dataset_listfile: str,
        valued_feats: List[str],
        event_feats: List[str],
        static_feats: List[str],
        text_feats: Optional[List[str]] = None,
        prediction_task: Optional[str] = 'all',
        phenotypes_listfile: Optional[str] = None,
        get_target_data: Optional[Callable] = None,
        n_examples: Optional[int] = None
    ):
        """Initialize the reader class.
        
        During initialization, the reader will read episode information from the dataset listfile and verify that all 
        patient IDs are present in the data directory. Patient-episode IDs are constructed as {pt_id}xxx, where xxx is 
        the episode number left-padded with zeros (max 3 digits).

        Args:
            dataset_listfile (str): Path to a CSV file containing episode file paths, patient IDs, and episode numbers.
            valued_feats (List[str]): A list of names of value-associated feature to extract from patient records.
            event_feats (List[str]): A list of names of event-associated features to extract from patient records.
            static_feats (List[str]): A list of names of static features to extract from patient records.
            prediction_task (Optional[str]): The prediction task the dataset will be used for. If the task is 
                'mortality', 'length_of_stay', or 'phenotype', the target data will be extracted using the appropriate 
                method. Otherwise, the user-defined method `self.get_target_data` will be called.
            get_target_data (Optional[Callable]): A user-defined method for extracting target data. This method should
                take an index as input and return the target data for that index. If `prediction_task` is not None, this
                method will be ignored.
            phenotypes_listfile (Optional[str]): Path to a CSV file containing phenotype data for patients. This file is
                created by running the `create_phenotypes.py`. If this file is not provided and `prediction_task` is 
                'phenotype' or 'all', an exception will be raised.
        """

        super().__init__()

        if prediction_task in ['phenotype', 'all']:
            if phenotypes_listfile is None:
                raise ValueError("phenotypes_listfile must be provided if prediction_task is 'phenotype' or 'all'.")
            if not os.path.exists(phenotypes_listfile):
                raise FileNotFoundError(f"Phenotypes listfile not found: {phenotypes_listfile}")
            self.phenotypes_listfile = phenotypes_listfile
        else:
            self.phenotypes_listfile = None
        
        self.dataset_listfile = dataset_listfile
        self.episode_file_paths = []
        self.patient_ids = []
        self.episode_numbers = []
        self.patient_episode_ids = []
        # Load the dataset file in a single operation
        if n_examples is not None:
            dataset_df = pd.read_csv(dataset_listfile, nrows=n_examples)
        else:
            dataset_df = pd.read_csv(dataset_listfile)
        self.episode_file_paths = dataset_df.iloc[:, 0].tolist()  # First column: file paths
        self.patient_ids = dataset_df.iloc[:, 1].astype(int).tolist()  # Second column: patient IDs
        self.episode_numbers = dataset_df.iloc[:, 2].astype(int).tolist()  # Third column: episode numbers
        # Calculate patient_episode_ids using vectorized operations
        self.patient_episode_ids = [pid * 1000 + enum for pid, enum in zip(self.patient_ids, self.episode_numbers)]
        self.data_root_path = os.path.commonpath(self.episode_file_paths)
        self.valued_feats = valued_feats
        self.event_feats = event_feats
        self.static_feats = static_feats
        self.text_feats = text_feats
        self.prediction_task = prediction_task
        if get_target_data is not None:
            self.get_target_data = get_target_data
        
        self._validate_patient_ids()  # Verify that all patient_ids are in data_root_path

    @staticmethod
    def get_pt_id_from_episode_id(episode_id: int) -> int:
        """Return the patient ID from a patient-episode ID."""
        return int(episode_id // 1e3)

    def get_stays_data(self, index: int) -> pd.DataFrame:
        """Load the patient's stay.csv data sorted by ICU INTIME"""
        pt_id = self.patient_ids[index]
        stay_data_path = Path(self.data_root_path) / str(pt_id) / 'stays.csv'
        stays_data = pd.read_csv(stay_data_path)
        stays_data = stays_data.sort_values('INTIME', ascending=True).reset_index(drop=True)
        return stays_data

    def get_mortality_target_data(self, index: int) -> int:
        """Get the in-hospital mortality status for a patient episode. If 1, patient died in hospital."""
        stays_data = self.get_stays_data(index)
        i = self.episode_numbers[index]
        mortality_status = stays_data['MORTALITY_INHOSPITAL'].iloc[i - 1]
        return mortality_status

    def get_length_of_stay_target_data(self, index: int) -> float:
        """Get the length of stay in hours for a patient episode."""
        stays_data = self.get_stays_data(index)
        i = self.episode_numbers[index]
        length_of_stay = stays_data['LOS'].iloc[i - 1] * 24  # Convert days to hours
        return length_of_stay

    def get_phenotype_target_data(self, index: int) -> List[int]:
        """Get the phenotype target data for a patient episode."""

        pheno_data = pd.read_csv(self.phenotypes_listfile, index_col=0)
        pheno_data = pheno_data.drop(columns=['period_length'])  # Length of stay is not needed for phenotyping
        
        pt_id =self.patient_ids[index]
        i = self.episode_numbers[index]

        for file_path in pheno_data.index:
            if str(pt_id) in file_path and f'episode{i}' in file_path:
                return pheno_data.loc[file_path].astype(int).tolist()
        
        raise ValueError(f"Phenotype data not found for patient ID {pt_id} and episode number {i}.")

    def get_feature_data(self, index: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get the feature data for a patient episode.
        
        Features are read from a single timeseries CSV file specified in the dataset listfile. Static features are read
        from stays.csv. Numeric and event features are extracted from the timeseries file based on self.numeric_feats
        and self.event_feats. Rows and columns containing only missing values are dropped from the feature dataframes.

        Args:
            index (int): The index of the patient episode ID in `self.patient_episode_ids`.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Static features, value-associated features, and 
                event-associated features for the patient episode.
        """
        
        episode_file_path = self.episode_file_paths[index]
        episode_ts_file_path = re.sub(r'.csv', '_timeseries.csv', episode_file_path)

        if self.static_feats:
            episode_df = pd.read_csv(episode_file_path)
            feat_cols = self._get_feature_column_names(self.static_feats, episode_df)
            static_data = episode_df.loc[:, feat_cols].iloc[0]
        else:
            static_data = None
            
        episode_ts_df = pd.read_csv(episode_ts_file_path, index_col='Hours')
        episode_ts_df.index = pd.to_timedelta(episode_ts_df.index, unit='h')
        

        if self.valued_feats:
            feat_cols = self._get_feature_column_names(self.valued_feats, episode_ts_df)
            val_data = episode_ts_df.loc[:, feat_cols]
            val_data = val_data.dropna(how='all')
        else:
            val_data = None

        if self.event_feats:
            feat_cols = self._get_feature_column_names(self.event_feats, episode_ts_df)
            event_data = episode_ts_df.loc[:, feat_cols]
            event_data = event_data.dropna(how='all')
        else:
            event_data = None

        if self.text_feats:
            feat_cols = self._get_feature_column_names(self.text_feats, episode_ts_df)
            text_data = episode_ts_df.loc[:, feat_cols]
            text_data = text_data.dropna(how='all')
        else:
            text_data = None
        
        return static_data, val_data, event_data, text_data

    def _get_target_method(self) -> Callable:
        if self.prediction_task is None:
            return self.get_target_data  # User-defined target method
        elif self.prediction_task == 'mortality':
            return self.get_mortality_target_data
        elif self.prediction_task == 'length_of_stay':
            return self.get_length_of_stay_target_data
        elif self.prediction_task == 'phenotype':
            return self.get_phenotype_target_data
        elif self.prediction_task == 'all':
            return (self.get_mortality_target_data, self.get_length_of_stay_target_data, self.get_phenotype_target_data)
        else:  # self.prediction_task is None
            t = self.prediction_task
            raise ValueError(f"prediction_task must be 'mortality', 'length_of_stay', 'phenotype', or None, got {t}.")
        
    def _get_feature_column_names(self, feature_names: List[str], df: pd.DataFrame) -> List[str]:
        """Get feature names from a DataFrame based on the provided base feature names.
        
        This method finds columns in the DataFrame that start with any of the base feature names. It is assumed that vector-valued features have one column per vector dimension, and the column names of vector-valued features are the base feature name followed by an underscore and the dimension index (e.g., 'feature_0', 'feature_1', etc.). Scalar features' column names are simply the base feature name without any suffix.

        Args:
            feature_names (List[str]): A list of base feature names to search for in the DataFrame.
            df (pd.DataFrame): The DataFrame to search for feature columns.
        
        Returns:
            List[str]: A list of column names in the DataFrame that match the base feature names.
        """

        feature_columns = []
        for base_name in feature_names:
            # Find columns that start with the base feature name
            matching_columns = [col for col in df.columns if re.search(f'^{re.escape(base_name)}(_\d+)?$', col)]
            if matching_columns:
                feature_columns.extend(matching_columns)
        return feature_columns
    
    def _validate_patient_ids(self) -> None:
        """Check that patient IDs are present in the root data directory."""
        dirs = os.listdir(self.data_root_path)
        for pt_id in self.patient_ids:
            if str(pt_id) not in dirs:
                raise ValueError(f"Patient ID {pt_id} not found in data directory.")

    def _convert_string_to_decimal_time(self, values: pd.Series) -> pd.Series:
        """Convert time values in HH:MM format to decimal format."""
        return values.str.split(':').apply(lambda t: float(t[0]) + float(t[1]) / 60)

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.patient_episode_ids)

    def __getitem__(self, index: int) -> Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
        record_id = self.patient_episode_ids[index]
        static_data, val_assoc_data, event_assoc_data, text_data = self.get_feature_data(index)
        if self.prediction_task != 'all':
            # Return one of mortality, length of stay, phenotype data, or a custom target
            targets = self._get_target_method()(index)
        else:
            # Return a list of (mortality status, length of stay, phenotypes) as targets
            targets = [fn(index) for fn in self._get_target_method()]
        return record_id, static_data, val_assoc_data, event_assoc_data, text_data, targets
    
    def index(self, patient_episode_id: int) -> int:
        """Get the index of a patient episode in the dataset."""
        return self.patient_episode_ids.index(patient_episode_id)
    