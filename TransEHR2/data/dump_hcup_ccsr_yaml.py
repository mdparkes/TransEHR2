"""
Maps ICD-10-CM codes to HCUP Clinical Classifications Software Refined (CCSR) categories using the 2024.1
release of the HCUP CCSR software. The mapping is dumped to a YAML file.

The 2024.1 release and other historic releases can be found at 
https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccsr_archive.jsp#ccsr.

The HCUP CCSR maps ICD-10-CM codes to one or more categories. This contrasts with ealier versions, i.e. CCS, which
mapped ICD codes to one and only one category by design. This was a necessary change to accureately represent 
information conveyed by ICD-10-CM codes.

The 2024.1 release of the HCUP CCSR contains default diagnosis categories for ICD-10-CM codes. Only one default
diagnosis category is assigned to each ICD-10-CM code. The default diagnosis category assumes that the ICD-10-CM
code is the principal diagnosis. There are two versions of default diagnosis: one for inpatients and one for
outpatients. 

The default diagnosis categories are not relevant for this project, as we are interested in identifying HCUP CCS/CCSR 
categories for all diagnoses, not just the principal diagnosis. Therefore, this script extracts all the non-default 
categories assigned to each ICD-10-CM code to populate the list of ICD-10-CM codes for each HCUP CCSR category.
"""

import os
import pandas as pd
import yaml

if __name__ == "__main__":

    # Load mappings from ICD-10-CM codes to CCSR categories
    file_path = os.path.join(os.path.dirname(__file__), "../../data/DXCCSR_v2024-1.csv")
    dxccsr = pd.read_csv(file_path, na_values=["''", "' '"], dtype=str)

    # Clean the enclosing single quotes from column names and values
    dxccsr.columns = dxccsr.columns.str.replace("'", "")
    dxccsr.replace(to_replace=r"'", value="", regex=True, inplace=True)
    dxccsr.set_index("ICD-10-CM CODE", inplace=True)


    ccsr_code_cols = slice(5, len(dxccsr.columns), 2)  # Non-default category codes start at column 5
    dxccsr.iloc[:, ccsr_code_cols] = dxccsr.iloc[:, ccsr_code_cols].replace({
        'RSP014': 'RSP011',  # Pneumothorax -> Pleurisy; pneumothorax; pulmonary collapse
    })
    ccsr_codes = dxccsr.iloc[:, ccsr_code_cols].to_numpy().ravel()

    ccsr_desc_cols = slice(6, len(dxccsr.columns), 2)  # Non-default category descriptions start at column 6
    # Replace CCSR respiratory condition descriptions in dataframe
    # Note that there will be some information corruption here, but this is a "best effort" solution to align CCS and
    # CCSR categories that are mostly the same. Here we replace CCSR labels with their closest CCS equivalents.
    dxccsr.iloc[:, ccsr_desc_cols] = dxccsr.iloc[:, ccsr_desc_cols].replace({
        'Acute hemorrhagic cerebrovascular disease': 'Acute cerebrovascular disease',
        'Diabetes mellitus with complication': 'Diabetes mellitus with complications',
        'External cause codes: complications of medical and surgical care, initial encounter': 'Complications of surgical procedures or medical care',
        'Other specified and unspecified liver disease': 'Other liver diseases',
        'Other specified and unspecified lower respiratory disease': 'Other lower respiratory disease',
        'Other specified and unspecified upper respiratory disease': 'Other upper respiratory disease',
        'Pleurisy, pleural effusion and pulmonary collapse': 'Pleurisy; pneumothorax; pulmonary collapse',
        'Pneumonia (except that caused by tuberculosis)': 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
        'Pneumothorax': 'Pleurisy; pneumothorax; pulmonary collapse',
        'Respiratory failure; insufficiency; arrest': 'Respiratory failure; insufficiency; arrest (adult)',
        'Septicemia': 'Septicemia (except in labor)'
    })
    ccsr_descriptions = dxccsr.iloc[:, ccsr_desc_cols].to_numpy().ravel()

    code_desc_pairs = pd.DataFrame({'CCSR CATEGORY': ccsr_codes, 'DESCRIPTION': ccsr_descriptions})
    code_desc_pairs = code_desc_pairs.dropna(axis=0, how='all').drop_duplicates()

    # Prepare the mapping of CCSR categories to ICD-10-CM codes
    output_dict = {}
    # Map CCSR category descriptions to CCSR category codes
    for _, (ccsr_code, desc) in code_desc_pairs.iterrows():
        output_dict[desc] = {
            'use_in_benchmark': False,
            'type': 'unknown',
            'id': ccsr_code,
            'codes': set()
        }
    # Map CCSR category description to ICD-10-CM codes
    for row in dxccsr.iterrows():
        icd_code = row[0]
        descriptions = row[1][ccsr_desc_cols].dropna().tolist()
        for desc in descriptions:
            output_dict[desc]['codes'].add(icd_code)
    # Convert the set of codes for each CCSR category to a list
    for key in output_dict.keys():
        output_dict[key]['codes'] = list(output_dict[key]['codes'])

    # Set the use_in_benchmark flag to True for select CCSR categories
    use_in_benchmark = [
        'Acute and unspecified renal failure',
        'Acute cerebrovascular disease',
        'Acute myocardial infarction',
        'Cardiac dysrhythmias',
        'Chronic kidney disease',
        'Chronic obstructive pulmonary disease and bronchiectasis',
        'Complications of surgical procedures or medical care',
        'Conduction disorders',
        'Coronary atherosclerosis and other heart disease',
        'Diabetes mellitus with complications',
        'Diabetes mellitus without complication',
        'Disorders of lipid metabolism',
        'Essential hypertension',
        'Fluid and electrolyte disorders',
        'Gastrointestinal hemorrhage',
        'Heart failure',
        'Hypertension with complications and secondary hypertension',
        'Other liver diseases',
        'Other lower respiratory disease',
        'Other upper respiratory disease',
        'Pleurisy; pneumothorax; pulmonary collapse',
        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
        'Respiratory failure; insufficiency; arrest (adult)',
        'Septicemia (except in labor)',
        'Shock'
    ]
    for desc in use_in_benchmark:
        output_dict[desc]['use_in_benchmark'] = True

    # Dump the mapping to a YAML file
    file_path = os.path.join(os.path.dirname(__file__), "../../data/hcup_ccsr_2024_definitions.yaml")
    with open(file_path, "w") as f:
        yaml.dump(output_dict, f, default_flow_style=False)
