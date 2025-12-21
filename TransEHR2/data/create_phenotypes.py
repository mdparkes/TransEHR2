from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import yaml
import random
import re

from tqdm import tqdm

random.seed(49297)


def process_partition(root_dir, output_dir, partition, file_table, definitions, eps=1e-6):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    icd9_code_to_group = {}
    for group in definitions:
        icd_codes = definitions[group]['icd9_codes']
        for code in icd_codes:
            if code not in icd9_code_to_group:
                icd9_code_to_group[code] = [group]
            else:
                icd9_code_to_group[code].append(group)

    icd10_code_to_group = {}
    for group in definitions:
        icd_codes = definitions[group]['icd10_codes']
        for code in icd_codes:
            if code not in icd10_code_to_group:
                icd10_code_to_group[code] = [group]
            else:
                icd10_code_to_group[code].append(group)

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

    xty_triples = []
    patients = file_table['patient_id'].unique()
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(root_dir, patient)
        episode_numbers = file_table[file_table['patient_id'] == patient]['episode']
        patient_ts_files = [f"episode{i}_timeseries.csv" for i in episode_numbers]

        for ts_filename in patient_ts_files:
            lb_filename = ts_filename.replace("_timeseries", "")
            label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
            
            # empty label file
            if label_df.shape[0] == 0:
                continue

            los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
            if pd.isnull(los):
                print("\n\t(length of stay is missing)", patient, ts_filename)
                continue

            ts_file_path = os.path.abspath(os.path.join(root_dir, patient, ts_filename))

            cur_labels = [0 for _ in range(len(id_to_group))]

            icustay = label_df['Icustay'].iloc[0]  
            diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                       dtype={"ICD_CODE": str, "ICD_VERSION": int})
            diagnoses_df = diagnoses_df[diagnoses_df.ICUSTAY_ID == icustay]
            for _, row in diagnoses_df.iterrows():
                if row['USE_IN_BENCHMARK']:
                    code = row['ICD_CODE']
                    if row['ICD_VERSION'] == 9:
                        try:
                            group = icd9_code_to_group[code]
                        except KeyError:
                            print(f"Code {code} not found in CCS category mappings for ICD9.")
                            continue

                    elif row['ICD_VERSION'] == 10:
                        try:
                            group = icd10_code_to_group[code]
                        except KeyError:
                            print(f"Code {code} not found in CCSR category mappings for ICD10.")
                            continue
                    else:
                        continue
                    for g in group:
                        group_id = group_to_id[g]
                        cur_labels[group_id] = 1

            cur_labels = [x for (i, x) in enumerate(cur_labels) if definitions[id_to_group[i]]['use_in_benchmark']]

            xty_triples.append((ts_file_path, los, cur_labels))

    if partition == "train":
        random.shuffle(xty_triples)
    if partition in ("val", "test"):
        xty_triples = sorted(xty_triples)

    codes_in_benchmark = [x for x in id_to_group if definitions[x]['use_in_benchmark']]

    listfile_header = "stay,period_length," + ",".join(codes_in_benchmark)
    with open(os.path.join(output_dir, f"phenotyping_{partition}_listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, t, y) in xty_triples:
            labels = ','.join(map(str, y))
            listfile.write('{},{:.6f},{}\n'.format(x, t, labels))


def main():
    parser = argparse.ArgumentParser(description="Create data for phenotype classification task.")
    parser.add_argument('root_path', type=str, help="Directory where all patients' timeseries are stored.")
    parser.add_argument('output_path', type=str, help="Directory where the created listfile should be stored. "
                        "Listfiles will be saved in a different subdirectory for each cross validation fold.")
    parser.add_argument('--listfile_dir', type=str, default=os.path.join(os.path.dirname(__file__), '../../data'),
                        help='Directory where CSV listfiles for each training/validation/test partition are located')
    parser.add_argument('--icd9_phenotype_definitions', '-p9', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../../data/hcup_ccs_2015_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    parser.add_argument('--icd10_phenotype_definitions', '-p10', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../../data/hcup_ccsr_2024_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.icd9_phenotype_definitions) as definitions_file:
        icd9_defs = yaml.safe_load(definitions_file)
    with open(args.icd10_phenotype_definitions) as definitions_file:
        icd10_defs = yaml.safe_load(definitions_file)

    definitions = dict()

    for group in icd9_defs:
        if group in definitions.keys():
            definitions[group]['icd9_codes'].extend(icd9_defs[group]['codes'])
            if not definitions[group]['use_in_benchmark']:  # If already True, keep it that way
                definitions[group]['use_in_benchmark'] = icd9_defs[group]['use_in_benchmark']
        else:
            definitions[group] = {
                'icd9_codes': icd9_defs[group]['codes'],
                'icd10_codes': [],
                'use_in_benchmark': icd9_defs[group]['use_in_benchmark']
            }

    for group in icd10_defs:
        if group in definitions.keys():
            definitions[group]['icd10_codes'].extend(icd10_defs[group]['codes'])
            if not definitions[group]['use_in_benchmark']:
                definitions[group]['use_in_benchmark'] = icd10_defs[group]['use_in_benchmark']
        else:
            definitions[group] = {
                'icd9_codes': [],
                'icd10_codes': icd10_defs[group]['codes'],
                'use_in_benchmark': icd10_defs[group]['use_in_benchmark']
            }

    partition_table_files = [f for f in os.listdir(args.listfile_dir) if f.startswith('fold')]
    n_folds = max([int(re.match(r'fold(\d+)', f).group(1)) for f in partition_table_files])
    
    for i in range(n_folds + 1):
        print(f"Processing fold {i}...")
        fold_path = os.path.join(args.output_path, f"fold{i}")
        train_file_table = pd.read_csv(
            os.path.join(args.listfile_dir, f'fold{i}_train.csv'), 
            dtype={'patient_id': str, 'episode': int}
        )
        test_file_table = pd.read_csv(
            os.path.join(args.listfile_dir, f'fold{i}_test.csv'),
            dtype={'patient_id': str, 'episode': int}
        )
        if os.path.exists(os.path.join(args.listfile_dir, f'fold{i}_val.csv')):
            val_file_table = pd.read_csv(
                os.path.join(args.listfile_dir, f'fold{i}_val.csv'),
                dtype={'patient_id': str, 'episode': int}
            )
            process_partition(args.root_path, fold_path, "val", val_file_table, definitions)
        process_partition(args.root_path, fold_path, "train", train_file_table, definitions)
        process_partition(args.root_path, fold_path, "test", test_file_table, definitions)
        print('\n')


if __name__ == '__main__':
    main()
