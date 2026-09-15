# TransEHR2

## About

TransEHR, originally presented by [Xu *et al.*](https://proceedings.mlr.press/v225/xu23a/xu23a.pdf), is a transformer neural network-based model that learns representations of medical record timeseries which can be used as input for downstream medical prediction tasks. Xu *et al.* used TransEHR to process medical records from the first 48 hours of patients' stays in ICU and predict their length of stay, in-hospital mortality, and International Classification of Disease (ICD) codes assigned during their stay.

TransEHR consists of a generator network, a discriminator network, and a transformer Hawkes process network. During self-supervised pre-training, the generator learns to simulate the values of randomly masked records. The discriminator network learns to identify which records are simulated and which ones are original. The transformer Hawkes process learns the temporal dynamics of different types of features captured in the medical records. TransEHR is pretrained to minimize the sum of losses from these three networks. Finetuning is fully supervised and aims to maximize performance on a given downstream prediction task.

TransEHR2 improves upon the original TransEHR model:

- It supports vector-valued, categorical and text features, where the original supported only scalar value-associated features.
- It distinguishes between records collected before and after a reference time, so it can use information that appears only in antecedent records, such as discharge summaries from previous hospitalizations.
- It is set up to work with MIMIC-IV, where the original was evaluated on MIMIC-III among other datasets.
- It corrects known errors in Xu *et al.*'s loss calculations for the transformer Hawkes process.
- It supports cross-validation, which was not implemented in Xu *et al.*'s code.

## Installation

Clone the repository and create a virtual environment (optional but advisable).

```shell
git clone https://github.com/mdparkes/TransEHR2.git && cd TransEHR2
python -m venv venv/TransEHR2
```

Install the required libraries. Note for high performance compute cluster users: you may have to build `transformer-engine` before installing other libraries from `requirements.txt` to use fp8 precision.

Otherwise, if you don't need fp8 precision, just do

```shell
source venv/TransEHR2/bin/activate
pip install -r requirements.txt
deactivate
```

The text encoder is named by `LLM_NAME` in `TransEHR2/constants.py`, along with the token limit and the pooling rule. It must be available to the machine that runs `embed_text.py`, and may be either a HuggingFace model id or a local path; the shipped default is a cluster-local copy of `BAAI/bge-m3`. `embed_text.py --llm-name` overrides it for one run. For a gated model, put a HuggingFace token in a `.env` file at the repository root as `HF_READ_TOKEN`.

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

## Repository layout

```
TransEHR2/
├── TransEHR2/                     Package: model, losses, training routines, data pipeline
│   ├── configs/
│   │   ├── datasets/              Dataset configs (paths, feature selection, sequence lengths)
│   │   └── experiments/           Tuning specs under tuning/; per-experiment configs are generated
│   ├── constants.py               Text encoder, token limit, pooling, device selection
│   ├── losses.py                  Generator, discriminator and Hawkes process losses
│   ├── models.py                  ELECTRA, MixedClassifier
│   ├── modules.py                 Value and event encoders, generator, discriminator, Hawkes process
│   ├── layers.py                  Attention, encoder layers, temporal position encodings
│   ├── routines_accelerate.py     Pretrain, finetune and evaluation loops
│   └── data/                      Dataset, collation, standardization
├── hp_tuning/                     Sweep specs, trial expansion, result ranking
├── reporting/                     Evaluation, statistics and table building
│   └── jmir/                      Publisher house style: number formatting, table layout
├── tests/                         Test suite (pytest)
├── scripts/                       Entry points; run them from the repository root
│   ├── extract_data.py            MIMIC-IV episodes -> tensorized arrays
│   ├── embed_text.py              Pre-embed text features with the frozen encoder
│   ├── run_experiment.py          Pretrain, finetune, evaluate (single GPU)
│   ├── generate_tuning_configs.py  Expand a tuning spec into one config per trial
│   ├── tuning_trial.py            Look one trial up in a manifest, for job arrays
│   ├── report_tuning_results.py   Rank a sweep's trials
│   ├── select_tuned_hyperparameters.py  Assemble the winning config (additive sweep)
│   ├── select_tuned_cell.py       Write the winning cell's config (factorial sweep)
│   ├── generate_finetune_grid.py  Finetuning grid or seed repeats over one shared encoder
│   ├── report_experiment_results.py  Tabulate finished runs by name pattern
│   ├── dump_finetuned_predictions.py  Per-fold prediction CSVs
│   ├── compute_charlson_index.py  Charlson index per episode, and the cohort manifest
│   ├── run_charlson_logistic_regression.py  Charlson mortality baseline, per fold
│   └── report_results_tables.py   Result tables, one document per cohort
└── experiment_descriptions.md     What each experiment number means
```

Everything in `scripts/` is run from the repository root, as `python scripts/<name>.py`.

The batch scripts (`SLURM/` and the shell drivers at the root) are **not tracked**: what submits a job carries the absolute paths of the machine it runs on, so each installation keeps its own. The `sbatch` commands below are from a working copy that has them, and a fresh clone will not.

The per-experiment configs under `TransEHR2/configs/experiments/` are not tracked either. They are written by `generate_revision_experiments.py` from the tuned configuration, so a config file stays the record of what a finished run actually ran.

## Tests

```shell
pytest tests/ TransEHR2/
```

`tests/` holds the probes for the entry points and the reporting; `TransEHR2/` holds the model's own. Two tests in `tests/test_phase2_infrastructure.py` train a small model and are marked `slow`; `-m "not slow"` skips them.

## Running an experiment

An experiment consists of pretraining, finetuning and evaluating on a test set. Scripts take a dataset config and an experiment config, both under `TransEHR2/configs/`; edit those to change parameters.

**1. Extract the prepared data.** Reads the episode files `mimic4dataprep` produced and writes tensorized arrays.

```shell
python scripts/extract_data.py TransEHR2/configs/datasets/mimic4.yaml
```

**2. Embed text**, if the experiment uses text features. Scans the extracted dataset directories and writes one embedding per note.

```shell
python scripts/embed_text.py --data-dir ${DATA_DIR}
```

**3. Train and evaluate.**

```shell
python scripts/run_experiment.py TransEHR2/configs/datasets/mimic4.yaml TransEHR2/configs/experiments/<experiment>.yaml
```

`--folds` restricts the run to particular folds and `--tasks` to particular tasks, which is how the work is spread across jobs. `fold0` is reserved for hyperparameter tuning and is excluded from reported results. `--tasks none` pretrains only.

`--num_workers` and `--mixed_precision` (default `bf16`) are the two flags worth setting per machine. `--limit_episodes` runs the whole cycle on a handful of episodes to check a configuration end to end; its results are meaningless as measurements and are written to the paths a real run would use, so point `MODEL_DIR` somewhere disposable. `--help` lists the rest.

### Re-running an experiment

The training scripts skip work whose output already exists, so a job that is requeued after a time limit resumes instead of starting over. The same behaviour means that re-running an experiment *deliberately* — after a code or configuration change — will silently reuse the previous run's results unless its outputs are cleared first.

Everything is keyed on `EXPERIMENT_NAME` from the experiment config, with `MODEL_DIR` from the same file. `checkpoints/` and `log/` are relative to the working directory the script is launched from.

| Artefact | Path | Effect on a re-run |
|---|---|---|
| Pretrained weights | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/pretrained/*.pt` | **Pretraining is skipped entirely.** Any `.pt` in the directory counts; the most recently written is loaded. |
| Finetuned weights | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/pretrained/finetuned_<task>.pt` | Finetuning is skipped for that task. Evaluation still runs. |
| Task evaluation | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/<task>/evaluation/evaluation_<task>.yaml` | That task is skipped entirely, finetuning and evaluation both. |
| Checkpoints | `checkpoints/<EXPERIMENT_NAME>/<fold>/{pretrained,finetuned_<task>}/` | Training resumes from the recorded epoch. **Removed automatically when a run completes**, so these are present only after a job was killed. |
| TensorBoard logs | `log/<EXPERIMENT_NAME>/<fold>/{pretrained,finetuned_<task>}/` | Nothing is skipped, but `SummaryWriter` appends: old and new curves are drawn together at overlapping steps. |
| Pretraining evaluation | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/pretrained/evaluation/evaluation_pretrained.yaml` | Overwritten. Stale only if pretraining was skipped. |

To re-run an experiment from scratch, remove its model tree and its logs:

```bash
rm -rf <MODEL_DIR>/<EXPERIMENT_NAME> log/<EXPERIMENT_NAME> checkpoints/<EXPERIMENT_NAME>
```

`--force_pretrain` and `--force_finetune` retrain without deleting anything, which is useful when the previous weights are still wanted. They are not equivalent to removing the tree: they overwrite the files a run writes, and leave any other `.pt` in the pretrained directory in place, where a later run that does not force will find it.

After a job is killed at its time limit, verify what it left before requeueing. A checkpoint is what makes the requeue resume, and a checkpoint from a superseded configuration makes it resume the wrong run:

```bash
ls checkpoints/
```

## Hyperparameter tuning

A sweep is described by a spec under `TransEHR2/configs/experiments/tuning/`. The default sweep is additive: an all-defaults centre per encoding arm, plus one trial for each non-default value of each hyperparameter. Each trial becomes a standalone experiment config that one job can run.

```shell
# Expand the spec into one config per trial, plus a manifest
python scripts/generate_tuning_configs.py TransEHR2/configs/experiments/tuning/<spec>.yaml

# Look up trial $SLURM_ARRAY_TASK_ID; prints the config path for run_experiment.py
python scripts/tuning_trial.py ${MANIFEST} --stage pretrain --index ${SLURM_ARRAY_TASK_ID} --field config

# Rank the results. Safe to run before every trial has finished.
python scripts/report_tuning_results.py ${MANIFEST} --progress

# Write the winning config
python scripts/select_tuned_hyperparameters.py ${MANIFEST} --arm ${ARM} --output ${CONFIG}
```

`tuning_trial.py` indexes the same lists the job arrays index, so `--stage` is required: the pretrain and finetune arrays cover different subsets of one manifest. `--count` gives the array bound, `--list` prints the whole stage.

Trials are ranked on the criterion each hyperparameter's grid entry names in the spec: `select_on: pretrain` ranks on pretraining loss, `select_on: mortality` on mortality validation performance.

For a factorial sweep, where trials are cells of a grid rather than one-at-a-time departures from a centre, `select_tuned_cell.py` takes the same arguments and writes the winning cell's config. `report_tuning_tables.py` builds the sweep's results as a table, reading evaluation YAMLs directly rather than a manifest, so it serves either sweep shape.

## Charlson comorbidity baseline

Experiment 29 is a logistic regression on age at admission, sex and the Charlson comorbidity index, reported against experiment 28 — a peri-stay-only model on the same episodes — as a conventional-severity-score reference for in-hospital mortality. The index is that of the patient's most recent earlier hospital admission.

**The cohort is an explicit episode manifest, and both arms are given the same file.** An episode belongs to the comparison exactly when all three features exist: an index, an age and a sex. That is not a predicate over the extracted arrays, because whether the index could be computed depends on `diagnoses.csv`. Both arms resolve the one file by patient-episode ID through the same loader, so they select identical episodes in identical order, which is what the paired test needs. Do not substitute the `diagnosis_history` cohort: it asks whether the *text* of an earlier admission's diagnoses survived extraction, which neither arm reads, and it excludes episodes whose codes are perfectly available.

```shell
# 1. Score the earlier admissions and write the cohort manifest. Reads stays.csv and
#    diagnoses.csv per subject, then intersects with the episodes that have an age and a sex.
#    Writes misc/charlson/charlson_index.csv and misc/charlson/charlson_cohort.txt, and fails
#    if any fold is short of a cohort episode.
python scripts/compute_charlson_index.py ${DATASET_CONFIG} -w 8 --write_cohort

# 2. Generate the configs. Experiment 28 records the manifest path in COHORT_EPISODES, so
#    write the manifest first; run_experiment.py refuses to start if the file is missing.
python scripts/generate_revision_experiments.py

# 3. Fit the regression: one fit per fold on that fold's training split, predicting its
#    validation and test splits, in the layout dump_finetuned_predictions.py uses.
python scripts/run_charlson_logistic_regression.py ${DATASET_CONFIG}

# 4. Train and dump the control, like any other experiment.
EXPERIMENT_CONFIG=TransEHR2/configs/experiments/experiment28_peristay_charlsonsubset_rev.yaml \
TASKS=mortality \
    sbatch --array=0-4 SLURM/slurm_run_experiment.sh
sbatch --array=0-0 SLURM/slurm_dump_predictions.sh \
    TransEHR2/configs/experiments/experiment28_peristay_charlsonsubset_rev.yaml

# 5. Build the table.
python scripts/report_results_tables.py --cohorts charlson
```

The fit is unpenalized, so it is plain maximum likelihood with no regularization strength to tune, and its decision threshold is calibrated on the validation split exactly as for every other arm. Per-fold coefficients and odds ratios are written to `charlson_coefficients.csv` in the experiment directory.

Age and sex are read from the extracted `static_data` array, so they are the values the deep models receive. Note that the extraction's `Age` is MIMIC-IV `anchor_age` carried onto every stay of a patient rather than an age recomputed at each admission.

## Reporting results

### Dumping predictions

The reporting scripts read per-fold prediction CSVs, not the aggregated `*_evaluation.yaml` files, because calibrating a decision threshold needs the raw predicted probabilities. Run the dump for every experiment that will appear in a table:

```shell
python scripts/dump_finetuned_predictions.py ${DATASET_CONFIG} ${EXPERIMENT_CONFIG} ${EXPERIMENT_NAME}
```

This produces one CSV per fold, task and split:

```
models/
└── experiment20_peristay_textsubset_rev/
    ├── fold1/
    │   ├── mortality/
    │   │   ├── mortality_train_finetuned_output.csv
    │   │   ├── mortality_val_finetuned_output.csv
    │   │   └── mortality_test_finetuned_output.csv
    │   ├── length_of_stay/
    │   │   └── ...
    │   └── phenotype/
    │       └── ...
    ├── fold2/
    └── ...
```

By default the `test` split supplies the reported numbers, and the `val` split is also required for the two classification tasks, because that is where the decision threshold is calibrated. Length of stay needs only `test`, and the reporting scripts never read the `train` split. Use `--split` to report a split other than `test`.

### Building tables

Experiment numbers are given in the order their columns should appear, left to right, and one is nominated as the control that every other column is tested against. Numbers are resolved by globbing `experiment{N}_*` under `--model-dir`, so `--experiments 20` finds `experiment20_peristay_textsubset_rev`. See `experiment_descriptions.md` for what each number is.

`report_results_tables.py` declares the cohorts — `textsubset`, `historysubset` and `charlson` — with their column order and the control each is tested against, and writes one document per cohort with a numbered table per task:

```shell
python scripts/report_results_tables.py
```

A task whose predictions have not been dumped is named and skipped rather than aborting the tables that are ready. Restrict the run with `--tasks` and `--cohorts`, and check a single table on screen by naming one of each.

Any other set of experiments can be reported by giving them explicitly, in the order their columns should appear:

```shell
python scripts/report_results_tables.py --experiments 23 22 20 --control 20 --tasks mortality
```

Options the script does not define are passed through to the per-task reporter, so the threshold, metric, fold and formatting flags below all still apply:

```shell
python scripts/report_results_tables.py --tasks mortality --threshold 0.5
```

The Word tables carry P values only. A stats CSV is written beside each one with every per-fold value, mean difference, *t* statistic, degrees of freedom, and both unadjusted and adjusted P value.

### Statistical comparisons

Models are compared with the corrected resampled *t* test of [Nadeau and Bengio (2003)](https://doi.org/10.1023/A:1024068626366), paired on folds. Because the training sets of the folds overlap, the per-fold estimates are correlated and their sample variance underestimates the variance of the mean difference; the correction inflates it by the ratio of test set size to training set size:

```
t = mean(d) / sqrt((1/n + n_test/n_train) * var(d, ddof=1))
```

on *n* − 1 degrees of freedom, where *d* holds the per-fold differences. For one run of *k*-fold cross-validation, `n_test/n_train = 1/(k − 1)`.

*k* is a property of the split, not of how many runs finished, so pass `--cv-folds` when a fold is missing from the report: a model still trained on (k−1)/k of the data, and taking *k* from the folds found would make a partial comparison conservative for a reason that has nothing to do with the data.

P values are adjusted with the Benjamini-Hochberg procedure. `--fdr-scope` selects the family: `table` (the default), `row`, or `none`.

### Decision thresholds

`--threshold` controls how predicted probabilities become hard class labels for the two classification tasks:

- **`prevalence`** (the default) — within each fold the threshold is set on the `--calibration-split` (default `val`) so that the predicted positive rate matches the observed prevalence, then applied unchanged to the reported split. The threshold is never chosen on the split being reported, and the same rule is applied to every model including the control. The thresholds used are recorded in a table footnote.
- **a fixed number**, e.g. `--threshold 0.5` — no calibration, and the `val` split is not needed.

For the multi-label diagnosis task, `--phenotype-threshold-scope per-label` (the default) calibrates each of the 25 labels independently. `global` shares one threshold across all labels, which is more conservative for the rarest labels, whose calibration split holds few positive examples.

AUROC and AUPRC are threshold-free and are unaffected by any of this.

### Option reference

`report_results_tables.py` defines `--tasks`, `--cohorts`, `--experiments`, `--control`, `--tables_dir` and `--dry_run`, and passes everything else through to the per-task reporter. `--help` on either lists the full set; the options below are the ones that change what is computed rather than how it is printed.

| Option | Default | Purpose |
|---|---|---|
| `--experiments N [N ...]` | required | Experiment numbers, in column order |
| `--control N` | required | Experiment every other column is tested against |
| `--model-dir DIR` | `./models` | Directory holding one subdirectory per experiment |
| `--split SPLIT` | `test` | Split to report |
| `--folds FOLD [FOLD ...]` | auto-discover | Restrict to a common set of folds |
| `--cv-folds N` | folds found | Folds the *design* has, which sets `n_test/n_train` in the correction. Not the number averaged over: a model still trains on (k−1)/k of the data when one fold's run is missing. |
| `--fdr-scope {table,row,none}` | `table` | Family for the Benjamini-Hochberg adjustment |
| `--metrics KEY [KEY ...]` | task default | Select and reorder rows (`--list-metrics` prints the keys) |

Classification tasks only (`--tasks mortality`, `--tasks phenotype`):

| Option | Default | Purpose |
|---|---|---|
| `--threshold {prevalence,FLOAT}` | `prevalence` | How probabilities become class labels |
| `--calibration-split SPLIT` | `val` | Split the threshold is calibrated on |
| `--phenotype-threshold-scope {per-label,global}` | `per-label` | One threshold per label, or one shared (`--tasks phenotype` only) |

Output and formatting — `--output`, `--append`, `--stats-csv`, `--precision`, `--alpha`, `--show-raw-p`, `--table-number`, `--caption`, `--labels`, `--quiet` — behave as their names suggest; see `--help`.

### Troubleshooting

| Message | Cause |
|---|---|
| `no fold directories found` | Predictions were not dumped, or `--model-dir` is wrong. The scripts cannot work from `*_evaluation.yaml`. |
| `Threshold calibration needs the 'val' split predictions` | The dump did not include the `val` split. Re-dump, or pass a fixed `--threshold`. |
| `the fold counts differ` | One experiment has more folds than another. Pass `--folds` to pick a common set. |
| `Refusing to calibrate the threshold on the split being reported` | `--calibration-split` matches `--split`. Calibrating on the reported split would bias the metrics. |
| `Experiment number N is ambiguous` | Two directories match `experimentN_*`. Rename one. |
| `Unknown metric(s): ...` | A `--metrics` key is not defined for that task. Use `--list-metrics` to see the valid keys. |
| `has no metric 'KEY'` | The dump for one experiment is incomplete, so a metric other experiments have is missing. Re-dump that experiment. |

## Distributed training is unsupported

`scripts/run_experiment_accelerate.py`, `scripts/tune_hyperparameters_accelerate.py` and the two `TransEHR2/configs/accelerate_config_*.yaml` files are present in the repository but are **not currently supported and should not be expected to work as-is.** They have not been kept in step with the single-GPU path, and no part of the workflow above depends on them.

Use `scripts/run_experiment.py`. Work is spread across GPUs by running independent jobs — one per fold, task, or configuration file — rather than by distributing a single run.
