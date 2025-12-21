import os
import yaml


if __name__ == "__main__":

    """Description of fields in var_map

    var_map is a mapping from feature names to properties of those features. Each feature is represented as a dictionary with the following keys:

        'type': 
        
            The data type of the feature, which can be 'numeric', 'categorical', or 'text'. This indicates the    original data type of the feature read from timeseries CSV files during data preprocessing. It is used to determine how the feature should be processed and what kind of loss to use for optimization during training.

        'size':

            The number of components in the feature. For numeric scalar features, this is 1. For numeric vector features, this is the number of dimensions in the vector. For categorical features, it is the number of unique categories. For text features, this is the number of strings that comprise the text feature (usually 1).

        'category_map':

            Only used for features of type 'categorical'. It maps integer values to categorical labels. The integer values should be 1-indexed, meaning that the first category is represented by 1, the second by 2, and so on. Zero is reserved to indicate that the categorical feature was not recorded at a particular timestep. If the data are already indexed from a different starting point, the data extraction tools will reindex them.
            
            If 'type' is not 'categorical', this key should map to an empty dictionary.
    
    """


    # In var_map, category_map is only used for categorical features. It maps integer values to categorical labels.
    # 'type' is either 'numeric', 'categorical', or 'text'. The 'numeric' type should not be confused with the numeric 
    # designation of the value-associated feature table. 'type' refers to the data type of the original feature read by 
    # the MimicDataReader class's __getitem__ method during data preprocessing, and is used to determine how the 
    # feature should be processed and what kind of loss to use during training.
    var_map = {
        # Features for numeric data tables
        'Diastolic arterial blood pressure':{
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'Glucose': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Heart rate': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'Mean arterial blood pressure': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'SpO2': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'Respiration rate': {            
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Systolic arterial blood pressure': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'Temperature': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'Height': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'Weight': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },

        # Features for event data tables
        'Albumin': {
            'type': 'numeric',
            'size': 1,
            'category_map': {} 
        },
        'ALP': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'ALT': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Arterial pCO2': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Arterial pH': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Arterial pO2': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'AST': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Bicarbonate': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Blood urea nitrogen': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Capillary refill rate': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Fraction inspired oxygen': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'GCS eye opening': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'GCS motor response': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'GCS verbal response': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Hematocrit': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Hemoglobin': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Lactate': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Lactate dehydrogenase': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Platelet count': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Serum creatinine': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Serum magnesium': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Serum sodium': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'SaO2': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Total bilirubin': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Total cholesterol': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Troponin I': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Troponin T': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Urine output': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Venous pCO2': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'Venous pO2': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },
        'White blood cell count': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        },

        # Features for text data tables
        'Discharge Summary': {
            'type': 'text',
            'size': 1,
            'category_map': {}
        },
        'Diagnosis Descriptions': {
            'type': 'text',
            'size': 1,
            'category_map': {}
        },

        # Static features:
        'Gender': {
            'type': 'categorical',
            'size': 1,
            'category_map': {0: 'Other', 1: 'F', 2: 'M'}
        },
        'Age': {
            'type': 'numeric',
            'size': 1,
            'category_map': {}
        }
    }

    # Dump var_map to a YAML file in /data
    root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(root, '..', '..', 'data', 'variable_properties.yaml')
    with open(output_path, 'w') as f_out:
        yaml.dump(var_map, f_out, default_flow_style=False, sort_keys=False)
