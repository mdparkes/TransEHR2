# TransEHR2

## About

TransEHR, originally presented by [Xu *et al.*](https://proceedings.mlr.press/v225/xu23a/xu23a.pdf), is a transformer neural network-based model that learns representations of medical record timeseries which can be used as input for downstream medical prediction tasks. Xu *et al.* used TransEHR to process medical records from the first 48 hours of patients' stays in ICU and predict their length of stay, in-hospital mortality, and International Classification of Disease (ICD) codes assigned during their stay.

TransEHR consists of a generator network, a discriminator network, and a transformer Hawkes process network. During self-supervised pre-training, the generator learns to simulate the values of randomly masked records. The discriminator network learns to identify which records are simulated and which ones are original. The transformer Hawkes process learns the temporal dynamics of different types of features captured in the medical records. TransEHR is pretrained to minimize the sum of losses from these three networks. Finetuning is fully supervised and aims to maximize performance on a given downstream prediction task.

TransEHR2 improves upon the original TransEHR model. It supports additional data types for input, namely: vector-valued features, categorical features, and text. In contrast, the original TransEHR model only supported scalar value-associated features. TransEHR2 also distinguishes between records collected before and after a reference time. For example, it can be set up to distinguish between medical records collected before and after admission to ICU. TransEHR can thus leverage information that only appears in antecedent records, such as discharge summaries from previous hospitalizations. Whereas TransEHR was originally evaluated on MIMIC-III data (among other datasets), TransEHR2 is set up to work with MIMIC-IV. TransEHR2 also corrects known errors in Xu *et al.*'s loss calculations for the transformer Hawkes process. It also supports cross-validation, which was not implemented in Xu *et al.*'s code.

TransEHR2 is potentially more computationally and memmory-intensive than its predecessor, particularly when it uses longer medical record histories and text features. Text features are embedded by an integrated Llama model with frozen weights. As such, the code is set up for multi-GPU computing. If text features are input to the model, fully sharded data-parallel computation is performed. Otherwise, distributed data parallel computing is done without model sharding.

## Installation

Clone the repository and create a virtual environment (optional but advisable).

```shell
git clone https://github.com/mdparkes/TransEHR2.git && cd TransEHR2
python -m venv venv/TransEHR2
```

Install the required libraries.

```shell
source venv/TransEHR2/bin/activate
pip install -r requirements.txt
deactivate
```

If you intend to use text features, you will require authorization to use Meta's Llama model. TransEHR2 uses HuggingFace to obtain the Llama module, and the exact version is specified in `TransEHR2/constants.py`. You must have an authorization token to use the model. TransEHR2 assumes that the authorization token is stored in a .env file at the root of the local repository. You will have to create this file with your own token.

### Installing optional libraries for MIMIC-IV data
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