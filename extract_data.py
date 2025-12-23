import argparse
import os
import re
import yaml

from TransEHR2.data.datareaders import MIMICDataReader
from TransEHR2.data.preprocessing import extract_mimic


def check_for_train_test_listfiles(fold_dir, fold_name):
    for partition in ['train', 'test']:
        dataset_listfile = os.path.join(fold_dir, f'{fold_name}_{partition}.csv')
        phenotypes_listfile = os.path.join(fold_dir, f'phenotyping_{partition}_listfile.csv')
        if not os.path.exists(dataset_listfile):
            raise FileNotFoundError(f"Missing listfile {dataset_listfile}")
        if not os.path.exists(phenotypes_listfile):
            raise FileNotFoundError(f"Missing listfile {phenotypes_listfile}")
        

def skip_validation(fold_dir, fold_name):
    dataset_listfile = os.path.join(fold_dir, f'{fold_name}_val.csv')
    phenotypes_listfile = os.path.join(fold_dir, f'phenotyping_val_listfile.csv')
    skip = False
    if not os.path.exists(dataset_listfile):
        print(f"Missing validation dataset listfile {dataset_listfile}.")
        skip = True
    if not os.path.exists(phenotypes_listfile):
        print(f"Missing validation phenotypes listfile {phenotypes_listfile}.")
        skip = True
    if skip:
        print("Skipping validation set.\n")
    return skip


if __name__ == "__main__":

    # Parse command line args
    parser = argparse.ArgumentParser(
        description="Extract data for all folds, prepare training, validation, and test datsets, and pickle them"
    )
    parser.add_argument(
        'dataset_config', type=str,
        help="YAML file that specifies parameters for data extraction"
    )
    parser.add_argument(
        '--n_examples', '-n', type=int, default=None,
        help="Number of examples to process. If not specified, all examples will be processed."
    )
    parser.add_argument(
        '--n_workers', '-w', type=int, default=1,
        help="Number of parallel worker processes to use for data extraction. Default is 1."
    )
    args = vars(parser.parse_args())
    n_examples = args['n_examples']
    with open(args['dataset_config'], 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    

    # Get parameters from config file(s)
    DATA_DIR = dataset_config['DATA_DIR']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    TEXT_FEATS = dataset_config['TEXT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']
    MAX_EPISODE_LEN_STEPS = dataset_config.get('MAX_EPISODE_LEN_STEPS', 500)
    MAX_HISTORY_LEN_STEPS = dataset_config.get('MAX_HISTORY_LEN_STEPS', 5000)
    MIN_EPISODE_LEN_STEPS = dataset_config.get('MIN_EPISODE_LEN_STEPS', 10)
    MIN_EPISODE_LEN_HOURS = dataset_config.get('MIN_EPISODE_LEN_HOURS', 48)
    MAX_EPISODE_LEN_HOURS = dataset_config.get('MAX_EPISODE_LEN_HOURS', 48)


    # Look for cross validation fold subdirectories
    fold_names = []
    for item in os.listdir(DATA_DIR):
        if re.match(r'fold\d+', item) and os.path.isdir(os.path.join(DATA_DIR, item)):
            fold_names.append(item)
    # Sort alphabetically
    fold_names.sort()

    # Search for listfiles for each fold
    for name in fold_names:
        fold_dir = os.path.join(DATA_DIR, name)
        check_for_train_test_listfiles(fold_dir, name)  # Raise an exception if listfiles not found
        for partition in ['train', 'test', 'val']:  # The validation partition is optional
            if partition == 'val' and skip_validation(fold_dir, name):
                continue  # Skip this fold if validation listfiles are missing
            dataset_listfile = os.path.join(fold_dir, f'{name}_{partition}.csv')
            phenotypes_listfile = os.path.join(fold_dir, f'phenotyping_{partition}_listfile.csv')

            print(f'Initializing datareader for {name}, {partition} set...')

            # MIMICDataReader loads raw data from CSV files into memory, but it still needs to be preprocessed.
            datareader = MIMICDataReader(
                dataset_listfile=dataset_listfile,
                phenotypes_listfile=phenotypes_listfile,
                valued_feats=VALUED_FEATS,
                event_feats=EVENT_FEATS,
                static_feats=STATIC_FEATS,
                text_feats=TEXT_FEATS,
                prediction_task='all',
                n_examples=n_examples
            )

            # Uses the datareader to get data into memory and internally uses a DataProcessor to preprocess it.
            # Creates a dataset object and pickles it on disk. The pickled dataset is read by the dataloader.
            extract_mimic(
                reader=datareader,
                suffix=partition,
                output_dir=fold_dir,
                max_episode_len_steps=MAX_EPISODE_LEN_STEPS,
                max_history_len_steps=MAX_HISTORY_LEN_STEPS,
                min_episode_len_steps=MIN_EPISODE_LEN_STEPS,
                min_episode_len_hours=MIN_EPISODE_LEN_HOURS,
                max_episode_len_hours=MAX_EPISODE_LEN_HOURS,
                n_workers=args['n_workers']
            )
