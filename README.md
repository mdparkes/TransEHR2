# TransEHR2

## About

TransEHR, originally presented by [Xu *et al.*](https://proceedings.mlr.press/v225/xu23a/xu23a.pdf), is a transformer neural network-based model that learns representations of medical record timeseries which can be used as input for downstream medical prediction tasks. Xu *et al.* used TransEHR to process medical records from the first 48 hours of patients' stays in ICU and predict their length of stay, in-hospital mortality, and International Classification of Disease (ICD) codes assigned during their stay.

TransEHR consists of a generator network, a discriminator network, and a transformer Hawkes process network. During self-supervised pre-training, the generator learns to simulate the values of randomly masked records. The discriminator network learns to identify which records are simulated and which ones are original. The transformer Hawkes process learns the temporal dynamics of different types of features captured in the medical records. TransEHR is pretrained to minimize the sum of losses from these three networks. Finetuning is fully supervised and aims to maximize performance on a given downstream prediction task.

TransEHR2 improves upon the original TransEHR model. It supports additional data types for input, namely: vector-valued features, categorical features, and text. In contrast, the original TransEHR model only supported scalar value-associated features. TransEHR2 also distinguishes between records collected before and after a reference time. For example, it can be set up to distinguish between medical records collected before and after admission to ICU. TransEHR can thus leverage information that only appears in antecedent records, such as discharge summaries from previous hospitalizations. Whereas TransEHR was originally evaluated on MIMIC-III data (among other datasets), TransEHR2 is set up to work with MIMIC-IV. TransEHR2 also corrects known errors in Xu *et al.*'s loss calculations for the transformer Hawkes process. It also supports cross-validation, which was not implemented in Xu *et al.*'s code.

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

### Re-running an experiment

The training scripts skip work whose output already exists, so a job that
is requeued after a time limit resumes instead of starting over. The same
behaviour means that re-running an experiment *deliberately* — after a
code or configuration change — will silently reuse the previous run's
results unless its outputs are cleared first.

Everything is keyed on `EXPERIMENT_NAME` from the experiment config, with
`MODEL_DIR` from the same file. `checkpoints/` and `log/` are relative to
the working directory the script is launched from.

| Artefact | Path | Effect on a re-run |
|---|---|---|
| Pretrained weights | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/pretrained/*.pt` | **Pretraining is skipped entirely.** Any `.pt` in the directory counts; the most recently written is loaded. |
| Finetuned weights | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/pretrained/finetuned_<task>.pt` | Finetuning is skipped for that task. Evaluation still runs. |
| Task evaluation | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/<task>/evaluation/evaluation_<task>.yaml` | That task is skipped entirely, finetuning and evaluation both. |
| Checkpoints | `checkpoints/<EXPERIMENT_NAME>/<fold>/{pretrained,finetuned}/` | Training resumes from the recorded epoch. **Removed automatically when a run completes**, so these are present only after a job was killed. |
| TensorBoard logs | `log/<EXPERIMENT_NAME>/<fold>/{pretrained,finetuned_<task>}/` | Nothing is skipped, but `SummaryWriter` appends: old and new curves are drawn together at overlapping steps. |
| Pretraining evaluation | `<MODEL_DIR>/<EXPERIMENT_NAME>/<fold>/pretrained/evaluation/evaluation_pretrained.yaml` | Overwritten. Stale only if pretraining was skipped. |

To re-run an experiment from scratch, remove its model tree and its logs:

```bash
rm -rf <MODEL_DIR>/<EXPERIMENT_NAME> log/<EXPERIMENT_NAME> checkpoints/<EXPERIMENT_NAME>
```

`--force_pretrain` and `--force_finetune` retrain without deleting
anything, which is useful when the previous weights are still wanted.
They are not equivalent to removing the tree: they overwrite the files a
run writes, and leave any other `.pt` in the pretrained directory in
place, where a later run that does not force will find it.

After a job is killed at its time limit, verify what it left before
requeueing. A checkpoint is what makes the requeue resume, and a
checkpoint from a superseded configuration makes it resume the
wrong run:

```bash
ls checkpoints/
```

### Before you start

**Predictions must already be dumped.** The reporting scripts read the
per-fold prediction CSVs written by `dump_finetuned_predictions.py`, not
the aggregated `*_evaluation.yaml` files, because calibrating a decision
threshold needs the raw predicted probabilities. Run the dump for every
experiment that will appear in a table:

```shell
python dump_finetuned_predictions.py ${DATASET_CONFIG} ${EXPERIMENT_CONFIG} ${EXPERIMENT_NAME}

# or, across several GPUs
accelerate launch dump_finetuned_predictions.py ${DATASET_CONFIG} ${EXPERIMENT_CONFIG} ${EXPERIMENT_NAME}
```

This produces one CSV per fold, task and split:

```
models/
└── experiment3_nohistory/
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

`fold0` is reserved for hyperparameter tuning and is always ignored. By
default the `test` split supplies the reported numbers, and the `val`
split is also required for the two classification tasks, because that is
where the decision threshold is calibrated. Length of stay needs only
`test`, and the reporting scripts never read the `train` split. Use
`--split` to report a split other than `test`.

**Install the dependencies.** Reporting adds `python-docx` to
`requirements.txt`:

```shell
pip install -r requirements.txt
```

### Step by step: tabulating results

**1. Decide the columns and the control.** Experiment numbers are given
in the order their columns should appear, left to right, and one of them
is nominated as the control that every other column is tested against.
Numbers are resolved by globbing `experiment{N}_*` under `--model-dir`,
so `--experiments 3` finds `experiment3_nohistory`. See
`experiment_descriptions.md` for what each number is.

The published tables use two column sets: experiments 3, 9, 1, 2, 7 for
the full dataset with experiment 3 as the control, and experiments 6, 4,
5 for the discharge summary subset with experiment 6 as the control.

**2. Check one table on screen before writing any files.** Omit
`--output` and nothing is written:

```shell
python report_mortality.py --experiments 3 9 1 2 7 --control 3
```

**3. Build all six tables into one document.** `--append` adds to an
existing document instead of replacing it:

```shell
OUT=tables/manuscript_tables.docx
rm -f $OUT

# Full dataset, control = experiment 3
python report_mortality.py       --experiments 3 9 1 2 7 --control 3 --table-number 1 --output $OUT --append
python report_length_of_stay.py  --experiments 3 9 1 2 7 --control 3 --table-number 2 --output $OUT --append
python report_phenotype.py       --experiments 3 9 1 2 7 --control 3 --table-number 3 --output $OUT --append

# Discharge summary subset, control = experiment 6
python report_mortality.py       --experiments 6 4 5 --control 6 --table-number 4 --output $OUT --append \
    --caption "In-hospital mortality prediction in the subset of patients with at least one discharge summary in their history."
python report_length_of_stay.py  --experiments 6 4 5 --control 6 --table-number 5 --output $OUT --append \
    --caption "Length-of-stay prediction in the subset of patients with at least one discharge summary in their history."
python report_phenotype.py       --experiments 6 4 5 --control 6 --table-number 6 --output $OUT --append \
    --caption "Diagnosis prediction in the subset of patients with at least one discharge summary in their history."
```

Delete the output file first, or `--append` will add a seventh table to
the six already there.

**4. Run the fixed-threshold sensitivity analysis.** Repeat the two
classification tasks with `--threshold 0.5` into a separate file. If the
ranking of experiments is unchanged, the conclusions do not rest on the
threshold rule, which is the answer a reviewer asking "why this
threshold?" actually needs:

```shell
python report_mortality.py --experiments 3 9 1 2 7 --control 3 \
    --threshold 0.5 --table-number 1 \
    --output tables/sensitivity_fixed_threshold.docx --append
python report_phenotype.py --experiments 3 9 1 2 7 --control 3 \
    --threshold 0.5 --table-number 3 \
    --output tables/sensitivity_fixed_threshold.docx --append
```

**5. Keep the full statistics.** The Word
table carries P values only. `--stats-csv` writes every per-fold value,
mean difference, *t* statistic, degrees of freedom, and both unadjusted
and adjusted P value:

```shell
python report_mortality.py --experiments 3 9 1 2 7 --control 3 \
    --stats-csv stats/table1_mortality.csv --quiet
```

### Statistical comparisons

Models are compared with the **corrected resampled *t* test** of [Nadeau
and Bengio (2003)](https://doi.org/10.1023/A:1024068626366), paired on folds, rather than a
Student *t* test. Because the training sets of the folds overlap, the
per-fold performance estimates are positively correlated and their sample
variance underestimates the variance of the mean difference. The
correction inflates that variance by the ratio of test set size to
training set size:

```
t = mean(d) / sqrt((1/n + n_test/n_train) * var(d, ddof=1))
```

on *n* − 1 degrees of freedom, where *d* holds the per-fold differences.
Without repeated folds, the **fixed adjustment** applies: for one run of
*k*-fold cross-validation, `n_test/n_train = 1/(k − 1)`.

P values are adjusted for multiple comparisons with the
Benjamini-Hochberg procedure. `--fdr-scope` selects the family: `table`
(the default: every comparison in the table), `row` (the comparisons
within one metric), or `none`. `--show-raw-p` reports the unadjusted
value alongside the adjusted one.

### Decision thresholds

`--threshold` controls how predicted probabilities become hard class
labels for the two classification tasks:

- **`prevalence`** (the default) — within each fold the threshold is set
  on the `--calibration-split` (default `val`) so that the predicted
  positive rate matches the observed prevalence, then applied unchanged
  to the reported split. The threshold is never chosen on the split being
  reported, and the same rule is applied to every model including the
  control. This keeps the predicted positive count honest and is stable
  to estimate for rare labels, since it inverts a quantile of a smooth
  distribution rather than maximising a jagged objective. The thresholds
  actually used are recorded in a table footnote.
- **a fixed number**, e.g. `--threshold 0.5` — no calibration, and the
  `val` split is not needed. Use it for the sensitivity analysis in
  step 4 above.

For the multi-label diagnosis task,
`--phenotype-threshold-scope per-label` (the default) calibrates each of
the 25 labels independently, which lifts the macro averages because a
single shared threshold is dominated by the common labels and the rare
ones rarely cross it. `--phenotype-threshold-scope global` shares one
threshold across all labels, which is more conservative for the rarest
labels, whose calibration split holds few positive examples.

AUROC and AUPRC are threshold-free and are unaffected by any of this.

### Option reference

Common to all three scripts:

| Option | Default | Purpose |
|---|---|---|
| `--experiments N [N ...]` | required | Experiment numbers, in column order |
| `--control N` | required | Experiment every other column is tested against |
| `--model-dir DIR` | `./models` | Directory holding one subdirectory per experiment |
| `--split SPLIT` | `test` | Split to report |
| `--folds FOLD [FOLD ...]` | auto-discover | Restrict to a common set of folds |
| `--metrics KEY [KEY ...]` | task default | Select and reorder rows |
| `--list-metrics` | off | Print the available metric keys and exit |
| `--precision N` | `3` | Decimal places for means and standard errors |
| `--alpha A` | `0.05` | Significance level, for the 3-decimal-place P value rule |
| `--fdr-scope {table,row,none}` | `table` | Family for the Benjamini-Hochberg adjustment |
| `--show-raw-p` | off | Report the unadjusted P value as well |
| `--table-number N` | `1` | Table number used in the caption |
| `--caption TEXT` | per task | Caption, without the "Table N." prefix |
| `--labels PATH` | `reporting_labels.yaml` | Column heading definitions |
| `--output PATH` | none | Word document to write |
| `--append` | off | Add to `--output` instead of replacing it |
| `--stats-csv PATH` | none | Per-fold values and test statistics |
| `--quiet` | off | Suppress the statistical detail block |

Classification tasks only (`report_mortality.py`, `report_phenotype.py`):

| Option | Default | Purpose |
|---|---|---|
| `--threshold {prevalence,FLOAT}` | `prevalence` | How probabilities become class labels |
| `--calibration-split SPLIT` | `val` | Split the threshold is calibrated on |

`report_phenotype.py` only:

| Option | Default | Purpose |
|---|---|---|
| `--phenotype-threshold-scope {per-label,global}` | `per-label` | One threshold per label, or one shared |

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
