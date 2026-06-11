# TransEHR2-IBDreadmission

This repository contains code for a machine learning model that predicts personalized readmission probability distributions for patients with inflammatory bowel diseases. It uses the TransEHR2 model framework learn latent representations of longitudinal medical records from these patients, and uses those representations to predict readimission probabilities.

## About TransEHR2

TransEHR, originally presented by [Xu *et al.*](https://proceedings.mlr.press/v225/xu23a/xu23a.pdf), is a transformer neural network-based model that learns representations of medical record timeseries which can be used as input for downstream medical prediction tasks. TransEHR consists of a generator network, a discriminator network, and a transformer Hawkes process network. During self-supervised pre-training, the generator learns to simulate the values of randomly masked records. The discriminator network learns to identify which records are simulated and which ones are original. The transformer Hawkes process learns the temporal dynamics of different types of features captured in the medical records. TransEHR is pretrained to minimize the weighted sum of losses from these three networks. Finetuning is fully supervised and aims to maximize performance on a given downstream prediction task.

TransEHR2 improves upon the original TransEHR model. It supports additional data types for input, namely: vector-valued, categorical, ordinal, and text features. TransEHR2 also corrects known errors in Xu *et al.*'s loss calculations for the transformer Hawkes process. The loss is reformulated to express the joint conditional likelihood of the next observed event type(s) their timestamp, rather than timestamp only, under the learned model.

## Data availability
The health records used to train and evaluate the model are restricted and, in the interest of protecting patient privacy, cannot be publicly distributed. Many of the supporting resources are used under license and cannot be redistributed here.

**TO DO: List sources for resources**

## Installation

Clone the repository and create a virtual environment (optional but advisable).

```shell
git clone https://github.com/phairlab/TransEHR2-IBDreadmission.git TransEHR2 && cd TransEHR2
python -m venv venv/TransEHR2
```

Install the required libraries. Note for high performance compute cluster users: you may have to build `transformer-engine` before installing other libraries from `requirements.txt` to use fp8 precision.

Otherwise, if you don't need fp8 precision, just do

```shell
source venv/TransEHR2/bin/activate
pip install -r requirements.txt
deactivate
```


If you intend to use text features, you will require authorization to use Meta's Llama model. TransEHR2 uses HuggingFace to obtain the Llama module, and the exact version is specified in `TransEHR2/constants.py`. You must have an authorization token to use the model. TransEHR2 assumes that the authorization token is stored in a .env file at the root of the local repository. You will have to create this file with your own token.

### Installing optional libraries for MIMIC-IV data
**TO DO: Rewrite this section for IBD data with IBDdataprep**
If you intend to use MIMIC-IV data with TransEHR2, install the MIMIC-IV data prep libraries. Create a separate virtual environment for the data prep library to avoid dependency conflicts (optional but advisable).

```shell
python -m venv venv/mimic4dataprep
source venv/mimic4dataprep/bin/activate
git clone https://github.com/mdparkes/mimic4dataprep/ ./mimic4dataprep
git clone https://github.com/mdparkes/datacleaner/ ./mimic4dataprep/datacleaner
pip install ./mimic4dataprep/datacleaner
pip install ./mimic4dataprep
deactivate
```

You will need to download the MIMIC-IV dataset and, optionally, the MIMIC-IV-Note dataset if you intend to use discharge summaries. Access to MIMIC-IV is credentialed. Authorized users can scrape the datasets with the following commands:

```shell
cd ${DATASET_DIR}
wget -r -N -c -np --user ${PHYSIONET_USERNAME} --ask-password https://physionet.org/files/mimiciv/3.1/
wget -r -N -c -np --user ${PHYSIONET_USERNAME} --ask-password https://physionet.org/files/mimic-iv-note/2.2/
```

See the `mimic4dataprep` documentation for instructions on how to extract the downloaded MIMIC-IV data.

## Using TransEHR2
TransEHR2 includes scripts for extracting MIMIC-IV data that has been prepared by `mimic4dataprep` (`extract_data.py`), hyperparameter tuning (`tune_hyperparameters_accelerate.py`), and executing experiments (`run_experiment_accelerate.py`). An experiment generally consists of pretraining, finetuning, and evaluating TransEHR2's performance on a test set. These scripts rely on configuration files in `TransEHR2/configs/`. Edit them to modify the scripts' parameters. Multi-GPU computing is facilitated by the `Accelerate` library and is configured by `accelerate_config_ddp.yaml` and `accelerate_config_fsdp.yaml`. Use FSDP when inputting text features to the model; this will shard the LLM module across GPUs to relieve memory pressure.
