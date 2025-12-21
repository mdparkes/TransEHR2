import numpy as np

from typing import Dict, List


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
