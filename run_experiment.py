"""
Single-GPU script for running experiments: pretrain, finetune, evaluate.

This is the single-GPU counterpart of ``run_experiment_accelerate.py``. It runs the same
training routines against the same configs; what it does not do is distribute. Phases 2 to 4
of the revision fan out by *configuration file* -- one independent job per hyperparameter
setting, per encoding arm, per fold -- which fills a node far better than 8-way DDP at batch
200, where each rank would see 25 samples and pay all-reduce on top.

Usage:
    python run_experiment.py <dataset_config> <experiment_config> [options]

    # One tuning trial: pretrain only, fold0, no downstream tasks
    python run_experiment.py TransEHR2/configs/datasets/mimic4.yaml \\
        TransEHR2/configs/experiments/tuning/p2_additive_lr_0.0002.yaml \\
        --folds fold0 --tasks none --num_workers 4

    # The finetune half of the same trial: reuses the pretrained encoders it just wrote
    python run_experiment.py TransEHR2/configs/datasets/mimic4.yaml \\
        TransEHR2/configs/experiments/tuning/p2_additive_lr_0.0002.yaml \\
        --folds fold0 --tasks mortality --num_workers 4

    # A manuscript run: every task, one fold per job
    python run_experiment.py TransEHR2/configs/datasets/mimic4.yaml \\
        TransEHR2/configs/experiments/experiment1_baseline.yaml --folds fold3

Differences from the accelerate version, all deliberate:

* **No distributed anything.** The Accelerator is still constructed -- ``routines_accelerate``
  is written against it and reducing to one process turns ``gather`` into the identity,
  ``wait_for_everyone`` into a no-op and ``is_main_process`` into True -- but the script
  refuses to start if it finds itself distributed. Launch with ``python``, never
  ``accelerate launch``.
* **Mixed precision is set here, in code.** ``accelerate launch`` was what read
  ``mixed_precision: bf16`` out of the accelerate config YAML. Nothing reads that file under a
  bare ``python`` launch, so bf16 is passed to the Accelerator explicitly and echoed at
  startup. Without this the run would silently be fp32, which is neither what the submitted
  experiments did nor what the single-GPU memory budget assumes.
* **``--folds`` and ``--tasks``.** The accelerate version hardcodes ``exclude='fold0'`` and
  loops over all three tasks. Phases 2 and 3 are fold0, mortality only; Phase 4 wants one fold
  per GPU.
* **Pretraining validation losses are persisted.** The accelerate version computes them and
  throws them away, which leaves no way to select a learning rate on pretraining loss.
"""

import argparse
import gc
import os
import re
import torch
import yaml

from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Union

import pickle

from TransEHR2.data.preprocessing import prepare_dataloaders
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import MaskedTokenDiscriminator, MaskedTokenGenerator, TransformerHawkesProcess
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder
from TransEHR2.routines_accelerate import pretrain_model, finetune_model, evaluate_finetuned_model
from TransEHR2.routines_accelerate import reshape_flattened_state_dict
from TransEHR2.utils import create_timer, convert_to_python_types, format_finetuning_performance_table, get_param_shapes


ALL_TASKS = ('mortality', 'length_of_stay', 'phenotype')

# Hyperparameters whose value the tuning report needs to read back off a finished run. Recorded
# in the evaluation YAML so a trial's result stays interpretable if the manifest that generated
# it is lost or edited.
RECORDED_HYPERPARAMETERS = (
    'POSITION_ENCODING',
    'PRETRAIN_LEARNING_RATE',
    'PRETRAIN_LEARNING_RATE_DECAY',
    'CMPNT_MASK_RATIO',
    'RECORD_MASK_RATIO',
    'THP_PRED_LOSS_TIME_WT',
    'OBS_UNOBS_SAMPLE_RATIO',
    'THP_LOSS_NLL_WEIGHT',
    'THP_PRED_LOSS_TYPE_WT',
    'BATCH_SIZE',
    'PRETRAIN_TOTAL_EPOCH',
    'FINETUNE_TOTAL_EPOCH',
    'HISTORY_LEN_STEPS',
    'EPISODE_LEN_STEPS',
    'USE_TEXT',
    'USE_HISTORICAL_RECORDS',
)


def initialize_accelerator(mixed_precision: str = 'bf16') -> Accelerator:
    """Build an Accelerator pinned to a single process.

    The Accelerator stays because ``TransEHR2.routines_accelerate`` is written against it, and
    on one process every collective it uses degrades to a no-op: ``gather`` returns its
    argument, ``wait_for_everyone`` returns immediately, ``is_main_process`` is True, and
    ``prepare`` moves the model to the GPU without a DDP or FSDP wrapper. Keeping it means the
    training code that runs here is byte-for-byte the code the correctness probes in
    ``TransEHR2/test_model_correctness.py`` were written against.

    What it does not do is protect against being launched the old way. ``accelerate launch``
    and a stray ``ACCELERATE_*`` variable in the environment both reach the Accelerator through
    env vars that this constructor cannot see, so the distributed state is asserted rather than
    assumed -- a silent fallback to 8 processes would run eight copies of the same trial over
    the same checkpoint directory.

    Args:
        mixed_precision: Passed straight to the Accelerator. 'bf16' matches what every previous
            experiment ran under via the accelerate config YAML, which a bare python launch
            does not read.

    Returns:
        Accelerator: A single-process Accelerator on the local GPU.

    Raises:
        RuntimeError: If the process finds itself in a distributed group.
    """
    accelerator = Accelerator(mixed_precision=mixed_precision)

    if accelerator.distributed_type != DistributedType.NO or accelerator.num_processes != 1:
        raise RuntimeError(
            f"run_experiment.py is the single-GPU entry point, but this process is "
            f"distributed_type={accelerator.distributed_type} across "
            f"{accelerator.num_processes} processes. Launch it with `python`, not "
            f"`accelerate launch`, and clear any ACCELERATE_* variables in the environment. "
            f"For a distributed run use run_experiment_accelerate.py instead."
        )

    return accelerator


def get_fold_names(data_dir: str, exclude: Optional[List[str]] = None) -> List[str]:
    """List the cross-validation fold directories under ``data_dir``.

    Args:
        data_dir: Directory holding the ``foldN`` subdirectories.
        exclude: Fold names to leave out. A bare string is accepted and treated as a
            single name -- the accelerate version passes ``exclude='fold0'`` into an
            ``item in exclude`` test, which works only because 'fold0' happens to be a
            substring of itself and would silently exclude nothing if the name were longer.

    Returns:
        The matching fold names, sorted.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    fold_names = []
    for item in os.listdir(data_dir):
        if item in exclude:
            continue
        if re.match(r'fold\d+$', item) and os.path.isdir(os.path.join(data_dir, item)):
            fold_names.append(item)
    fold_names.sort()
    return fold_names


def resolve_folds(data_dir: str, requested: Optional[List[str]]) -> List[str]:
    """Turn the ``--folds`` argument into a list of fold names that exist.

    Args:
        data_dir: Directory holding the fold subdirectories.
        requested: Fold names asked for, or None for "every fold except fold0".

    Returns:
        The fold names to run, in the order given (sorted when defaulted).

    Raises:
        ValueError: If a requested fold has no directory. Failing here costs a second; failing
            after a pretrain does not.
    """
    available = get_fold_names(data_dir, exclude=None)
    if not requested:
        # fold0 is the tuning fold and is held out of the manuscript results.
        return [name for name in available if name != 'fold0']

    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"Requested fold(s) {missing} not found in {data_dir}. Available: {available}"
        )
    return list(requested)


def resolve_tasks(requested: Optional[List[str]]) -> List[str]:
    """Turn the ``--tasks`` argument into a list of downstream tasks.

    Args:
        requested: Task names, ``['none']`` for pretraining only, or None for all three.

    Returns:
        The tasks to finetune and evaluate, possibly empty.

    Raises:
        ValueError: If a requested task is not one this codebase knows how to finetune.
    """
    if requested is None:
        return list(ALL_TASKS)
    if len(requested) == 1 and requested[0] == 'none':
        return []

    unknown = [task for task in requested if task not in ALL_TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}. Choose from {list(ALL_TASKS)} or 'none'.")
    return list(requested)


def truncate_loader(loader: torch.utils.data.DataLoader, n_episodes: int) -> torch.utils.data.DataLoader:
    """Rebuild a dataloader over only the first ``n_episodes`` episodes of its dataset.

    Smoke-test support. Truncating the *dataset* rather than breaking out of the batch loop
    keeps every epoch boundary, checkpoint write and early-stopping counter behaving exactly as
    it does in a real run, which is the point of the exercise: the test is of the plumbing, not
    of the model.

    The collate function is reused rather than rebuilt because it is a partial bound to the
    dataset's post-crop history length, and a Subset keeps the underlying dataset intact.

    Args:
        loader: The dataloader to shorten.
        n_episodes: How many episodes to keep. Larger than the dataset is a no-op.

    Returns:
        A new dataloader over the truncated dataset, or the original if no truncation applies.
    """
    if n_episodes is None or n_episodes >= len(loader.dataset):
        return loader

    subset = torch.utils.data.Subset(loader.dataset, list(range(n_episodes)))
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor if loader.num_workers > 0 else None,
        persistent_workers=loader.num_workers > 0,
        multiprocessing_context='spawn' if loader.num_workers > 0 else None
    )


def get_model_weights(dir: str) -> Union[dict, None]:
    """Load the most recently written model state dict in ``dir``, creating ``dir`` if absent.

    Args:
        dir: Directory to search for saved weights.

    Returns:
        The state dict of the most recently saved model, or None if there is none.
    """
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        return None

    saved_files = [
        os.path.join(dir, file) for file in os.listdir(dir)
        if file.endswith('.pt') and os.path.isfile(os.path.join(dir, file))
    ]
    if not saved_files:
        return None

    most_recent_file = max(saved_files, key=os.path.getctime)
    print(f"Loading most recently trained model's optimized weights from {most_recent_file}\n")
    return torch.load(most_recent_file, map_location='cpu', weights_only=False)


def write_pretrain_evaluation(
        evaluation_fp: str,
        experiment_name: str,
        fold_name: str,
        train_losses: Dict[str, Any],
        val_losses: Dict[str, Any],
        experiment_config: Dict[str, Any]
) -> None:
    """Persist the best-epoch pretraining losses.

    The accelerate version computes these and discards them, so learning rate and decay -- the
    two hyperparameters the revision plan selects on *pretraining* loss rather than on a
    downstream metric -- have nothing to be selected on. This writes them beside the
    per-task evaluation YAMLs, in the same shape, so one reporting script can read both.

    Args:
        evaluation_fp: Destination YAML path.
        experiment_name: Name of the experiment, echoed into the file.
        fold_name: Fold the losses come from.
        train_losses: Best-epoch training losses.
        val_losses: Best-epoch validation losses. ``Optimization_Loss`` is the selection metric.
        experiment_config: The full experiment config, from which the recorded hyperparameters
            are copied so the file stays interpretable on its own.
    """
    os.makedirs(os.path.dirname(evaluation_fp), exist_ok=True)
    evaluation_data = {
        'task': 'pretrain',
        'fold': fold_name,
        'experiment': experiment_name,
        # convert_to_python_types takes a mapping, so the dict is built first and converted
        # once. The values come straight out of YAML and are already native, but this keeps
        # the block identical in shape to the loss and score blocks beside it.
        'hyperparameters': convert_to_python_types({
            key: experiment_config[key]
            for key in RECORDED_HYPERPARAMETERS if key in experiment_config
        }),
        'train_losses': convert_to_python_types(train_losses),
        'val_losses': convert_to_python_types(val_losses),
    }
    with open(evaluation_fp, 'w') as f_out:
        yaml.dump(evaluation_data, f_out, default_flow_style=False, indent=2)
    print(f"Saved pretraining evaluation results to {evaluation_fp}\n")


def build_value_encoder(config: Dict[str, Any], prefix: str, n_features: int, feat_dim: int) -> ValueDataEncoder:
    """Construct a ValueDataEncoder from the ``prefix``-named keys of the experiment config.

    Both encoders and the downstream predictor's value encoder take the same eleven arguments
    from differently prefixed config keys. Building them through one function is what keeps the
    temporal ladder from being threaded to four of five construction sites -- the failure mode
    that ``TransEHR2/test_rope_encoding.py`` parses the entry points to catch.

    Args:
        config: The experiment config.
        prefix: 'GENERATOR_ENCODER' or 'DISCRIMINATOR_ENCODER'.
        n_features: Number of value features, text included when text is in use.
        feat_dim: Total flattened width of those features.

    Returns:
        The configured encoder.
    """
    return ValueDataEncoder(
        n_features=n_features,
        feat_dim=feat_dim,
        d_model=config[f'{prefix}_D_MODEL'],
        n_heads=config[f'{prefix}_N_HEADS'],
        n_encoder_blocks=config[f'{prefix}_N_ENCODER_BLOCKS'],
        dim_feedforward=config[f'{prefix}_DIM_FEEDFORWARD'],
        dropout=config[f'{prefix}_DROPOUT'],
        activation=config[f'{prefix}_ACTIVATION'],
        norm=config[f'{prefix}_NORM'],
        normalize_before=config.get(f'{prefix}_NORM_FIRST', True),
        position_encoding=config.get('POSITION_ENCODING', 'additive'),
        ladder_p_min=config.get('VALUE_LADDER_P_MIN', None),
        ladder_p_max=config.get('VALUE_LADDER_P_MAX', None)
    )


def build_event_encoder(config: Dict[str, Any], n_event_types: int) -> EventDataEncoder:
    """Construct an EventDataEncoder from the THP_ENCODER keys of the experiment config.

    Args:
        config: The experiment config.
        n_event_types: Number of event types the Hawkes process models.

    Returns:
        The configured encoder. Note that ``thp_encoder`` and ``predictor_event_encoder`` are
        separate instantiations rather than a shared object, so both go through here and both
        get the event ladder.
    """
    return EventDataEncoder(
        num_types=n_event_types,
        d_model=config['THP_ENCODER_D_MODEL'],
        d_inner=config['THP_ENCODER_D_INNER'],
        n_layers=config['THP_ENCODER_N_LAYERS'],
        n_head=config['THP_ENCODER_N_HEADS'],
        d_k=config['THP_ENCODER_D_K'],
        d_v=config['THP_ENCODER_D_V'],
        dropout=config['THP_ENCODER_DROPOUT'],
        normalize_before=config.get('THP_ENCODER_NORM_FIRST', True),
        position_encoding=config.get('POSITION_ENCODING', 'additive'),
        ladder_p_min=config.get('EVENT_LADDER_P_MIN', None),
        ladder_p_max=config.get('EVENT_LADDER_P_MAX', None)
    )


def main():
    """Run the experiment described by the two config files."""

    parser = argparse.ArgumentParser(
        description='Pretrain, finetune and evaluate a TransEHR2 model on a single GPU'
    )
    parser.add_argument(
        'dataset_config', type=str,
        help='YAML file that specifies parameters for the dataset'
    )
    parser.add_argument(
        'experiment_config', type=str,
        help='YAML file that specifies parameters for the experiment'
    )
    parser.add_argument(
        '--folds', type=str, nargs='+', default=None, metavar='FOLD',
        help='Folds to run, e.g. --folds fold0, or --folds fold1 fold2. Defaults to every '
             'fold except fold0, which is held out for tuning.'
    )
    parser.add_argument(
        '--tasks', type=str, nargs='+', default=None, metavar='TASK',
        help="Downstream tasks to finetune and evaluate: any of mortality, length_of_stay, "
             "phenotype. Pass 'none' to pretrain only. Defaults to all three."
    )
    parser.add_argument(
        '--force_pretrain', action='store_true',
        help='Pretrain even if pretrained weights are found'
    )
    parser.add_argument(
        '--force_finetune', action='store_true',
        help='Finetune and re-evaluate even if a finetuned model or an evaluation YAML exists'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Number of worker processes for data loading. Default is 0 (main process only).'
    )
    parser.add_argument(
        '--mixed_precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'],
        help='Mixed precision mode. Default bf16, which is what the accelerate config used.'
    )
    parser.add_argument(
        '--mem_test_mode', action='store_true',
        help='Run a single forward and backward pass, print peak memory, and terminate. Use '
             'this before committing to single-GPU packing for a configuration.'
    )
    parser.add_argument(
        '--limit_episodes', type=int, default=None, metavar='N',
        help='SMOKE TEST ONLY. Use only the first N episodes of each partition, so a full '
             'pretrain-finetune-evaluate cycle finishes in minutes. Results from a limited '
             'run are meaningless as measurements and are written to the same paths a real '
             'run would use, so point MODEL_DIR somewhere disposable.'
    )
    args = parser.parse_args()

    force_pretrain = args.force_pretrain
    force_finetune = args.force_finetune
    num_workers = args.num_workers
    mem_test_mode = args.mem_test_mode

    # One accelerator for the whole run. The accelerate version rebuilds it constantly
    # ("because it solves problems") -- those problems are process-group and FSDP-wrapper
    # lifetime issues that do not exist with one process and no wrapper.
    accelerator = initialize_accelerator(args.mixed_precision)

    with open(args.dataset_config, 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    TEXT_FEATS = dataset_config['TEXT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']
    # Extraction-time capacity of the history region, used to interpret the stored layout of
    # datasets extracted before that value was recorded in metadata.pkl.
    MAX_HISTORY_LEN_STEPS = dataset_config.get('MAX_HISTORY_LEN_STEPS', 0)

    with open(args.experiment_config, 'r') as f_in:
        experiment_config = yaml.safe_load(f_in)
    EXPERIMENT_NAME = experiment_config['EXPERIMENT_NAME']
    BATCH_SIZE = experiment_config['BATCH_SIZE']
    USE_TEXT = experiment_config['USE_TEXT']
    PREDICT_INDICATORS = experiment_config['PREDICT_INDICATORS']
    PREDICTOR_AGGREGATION_METHOD = experiment_config['PREDICTOR_AGGREGATION_METHOD']
    MODEL_DIR = experiment_config['MODEL_DIR']
    PRETRAIN_LEARNING_RATE = experiment_config.get('PRETRAIN_LEARNING_RATE', 2e-3)
    PRETRAIN_LEARNING_RATE_DECAY = experiment_config.get('PRETRAIN_LEARNING_RATE_DECAY', 0.9)
    PRETRAIN_TOTAL_EPOCH = experiment_config.get('PRETRAIN_TOTAL_EPOCH', 1000)
    DISC_LOSS_WEIGHT = experiment_config.get('DISC_LOSS_WEIGHT', 1.0)
    THP_LOSS_NLL_WEIGHT = experiment_config.get('THP_LOSS_NLL_WEIGHT', 1e-2)
    THP_LOSS_MC_SAMPLES = experiment_config.get('THP_LOSS_MC_SAMPLES', 100)
    USE_THP_PRED_LOSS = experiment_config.get('USE_THP_PRED_LOSS', True)
    THP_PRED_LOSS_TYPE_WT = experiment_config.get('THP_PRED_LOSS_TYPE_WT', 1.0)
    THP_PRED_LOSS_TIME_WT = experiment_config.get('THP_PRED_LOSS_TIME_WT', 1e-6)
    RECORD_MASK_RATIO = experiment_config.get('RECORD_MASK_RATIO', 0.15)
    OBS_UNOBS_SAMPLE_RATIO = experiment_config.get('OBS_UNOBS_SAMPLE_RATIO', 5.0)
    CMPNT_MASK_RATIO = experiment_config.get('CMPNT_MASK_RATIO', 0.25)
    FINETUNE_LEARNING_RATE = experiment_config.get('FINETUNE_LEARNING_RATE', 2e-4)
    FINETUNE_TOTAL_EPOCH = experiment_config.get('FINETUNE_TOTAL_EPOCH', 500)
    FINETUNE_LEARNING_RATE_DECAY = experiment_config.get('FINETUNE_LEARNING_RATE_DECAY', 0.9)
    USE_HISTORICAL_RECORDS = experiment_config.get('USE_HISTORICAL_RECORDS', True)
    # Runtime sequence-length caps. None uses everything that was extracted; smaller values crop
    # at load time, which is equivalent to re-extracting with the shorter limit.
    HISTORY_LEN_STEPS = experiment_config.get('HISTORY_LEN_STEPS', None)
    EPISODE_LEN_STEPS = experiment_config.get('EPISODE_LEN_STEPS', None)

    # A tuned hyperparameter's value in a tuning config is a one-element list, because that is
    # the shape HYPERPARAMETERS_TO_TUNE implies. Reject it loudly here rather than letting a
    # list reach an optimizer as a learning rate.
    if 'HYPERPARAMETERS_TO_TUNE' in experiment_config:
        raise ValueError(
            f"{args.experiment_config} carries HYPERPARAMETERS_TO_TUNE, so it is a tuning "
            f"grid rather than a single configuration. Expand it into one config per trial "
            f"with generate_tuning_configs.py and run those."
        )

    fold_name_list = resolve_folds(DATA_DIR, args.folds)
    task_list = resolve_tasks(args.tasks)

    print("=" * 70)
    print(f"Experiment:       {EXPERIMENT_NAME}")
    print(f"Device:           {accelerator.device}  (single process, "
          f"mixed_precision={accelerator.mixed_precision})")
    print(f"Folds:            {', '.join(fold_name_list)}")
    print(f"Tasks:            {', '.join(task_list) if task_list else 'none (pretrain only)'}")
    print(f"Encoding arm:     {experiment_config.get('POSITION_ENCODING', 'additive')}")
    print(f"Batch size:       {BATCH_SIZE}")
    print("=" * 70)

    timer = create_timer(
        results_dir=f'./log/timing/{EXPERIMENT_NAME}',
        experiment_name=EXPERIMENT_NAME
    )
    timer.start_total_timing()

    # Get the number of valued features and their sizes, the class counts of categorical
    # features, and the number of event types. These are model initialization arguments.
    with open(VARIABLE_PROPERTIES_PATH, 'r') as f_in:
        variable_properties = yaml.safe_load(f_in)
    tot_val_feat_dim = 0  # Total number of dimensions across all input features
    numeric_feat_dims = []  # The dimension of each numeric feature
    categorical_class_cnts = []  # The number of classes for each categorical feature
    ordinal_features = []  # The number of levels for each ordinal feature
    multilabel_class_cnts = []  # The number of classes for each multilabel feature
    for feature in VALUED_FEATS:
        tot_val_feat_dim += variable_properties[feature]['size']
        if variable_properties[feature]['type'] == 'numeric':
            numeric_feat_dims.append(variable_properties[feature]['size'])
        elif variable_properties[feature]['type'] == 'categorical':
            categorical_class_cnts.append(len(variable_properties[feature]['category_map']))
        elif variable_properties[feature]['type'] == 'ordinal':
            ordinal_features.append(len(variable_properties[feature]['category_map']))
        elif variable_properties[feature]['type'] == 'multilabel':
            multilabel_class_cnts.append(variable_properties[feature]['size'])

    if USE_TEXT:
        n_val_feats = len(VALUED_FEATS) + len(TEXT_FEATS)
        # text_embed_dim is a property of the extraction, so read it from the fold actually
        # being run rather than from whichever fold happens to sort first.
        meta_path = os.path.join(DATA_DIR, fold_name_list[0], 'train', 'metadata.pkl')
        with open(meta_path, 'rb') as f:
            _meta = pickle.load(f)
        text_embed_dim = _meta['text_embed_dim']
        if text_embed_dim == 0:
            raise RuntimeError(
                "text_embed_dim is 0 in dataset metadata. "
                "Run embed_text.py to pre-compute text embeddings before training."
            )
        tot_val_feat_dim += len(TEXT_FEATS) * text_embed_dim
        print(f"Text embedding dimension: {text_embed_dim}\n")
    else:
        n_val_feats = len(VALUED_FEATS)
        text_embed_dim = 0
    n_event_types = len(EVENT_FEATS)

    for fold_name in fold_name_list:

        timer.start_fold(fold_name)

        fold_dir = os.path.join(DATA_DIR, fold_name)
        dataloader_list = prepare_dataloaders(
            fold_dir,
            BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else 1,
            # Text balancing exists to stop one rank in a distributed run receiving all the
            # text-heavy episodes. With one process there is nothing to balance against, and
            # prepare_dataloaders skips the sampler at world_size 1 regardless.
            balance_text=False,
            use_historical_records=USE_HISTORICAL_RECORDS,
            history_len_steps=HISTORY_LEN_STEPS,
            episode_len_steps=EPISODE_LEN_STEPS,
            extracted_history_len_steps=MAX_HISTORY_LEN_STEPS
        )
        if len(dataloader_list) == 3:
            train_loader, val_loader, test_loader = dataloader_list
        else:
            train_loader, test_loader = dataloader_list[0], dataloader_list[-1]
            val_loader = None
        if args.limit_episodes is not None:
            dataloader_list = [
                truncate_loader(loader, args.limit_episodes) for loader in dataloader_list
            ]
            if len(dataloader_list) == 3:
                train_loader, val_loader, test_loader = dataloader_list
            else:
                train_loader, test_loader = dataloader_list[0], dataloader_list[-1]
            print(
                f"SMOKE TEST: every partition truncated to at most "
                f"{args.limit_episodes} episodes. Nothing this run produces is a measurement."
            )

        if task_list and val_loader is None:
            # finetune_model selects the best epoch on the validation set and would fail on
            # the first epoch, after paying for it. The extraction makes a val split, so this
            # means the fold was extracted without --make_val_set.
            raise RuntimeError(
                f"{fold_dir} has no val/ partition, so the tasks {task_list} cannot be "
                f"finetuned: there is nothing to select the best epoch on. Re-run the "
                f"partitioning step with --make_val_set, or pass --tasks none."
            )

        model_save_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/pretrained'
        pretrain_evaluation_fp = f'{model_save_dir}/evaluation/evaluation_pretrained.yaml'

        # ---------------------------------------------------------------- pretraining

        electra = ELECTRA(
            generator=MaskedTokenGenerator(
                encoder=build_value_encoder(
                    experiment_config, 'GENERATOR_ENCODER', n_val_feats, tot_val_feat_dim
                ),
                d_model=experiment_config['GENERATOR_D_MODEL'],
                numeric_dims=numeric_feat_dims,
                categorical_classes=categorical_class_cnts,
                ordinal_features=ordinal_features if ordinal_features else None,
                multilabel_classes=multilabel_class_cnts if multilabel_class_cnts else None,
                n_text_features=len(TEXT_FEATS) if USE_TEXT else 0,
                text_embed_dim=text_embed_dim,
                predict_indicators=PREDICT_INDICATORS,
                dim_feedforward=experiment_config['GENERATOR_DIM_FEEDFORWARD']
            ),
            discriminator=MaskedTokenDiscriminator(
                encoder=build_value_encoder(
                    experiment_config, 'DISCRIMINATOR_ENCODER', n_val_feats, tot_val_feat_dim
                ),
                d_model=experiment_config['DISCRIMINATOR_ENCODER_D_MODEL'],
                n_numeric_features=len(numeric_feat_dims),
                n_categorical_features=len(categorical_class_cnts),
                n_ordinal_features=len(ordinal_features),
                n_multilabel_features=len(multilabel_class_cnts),
                n_text_features=len(TEXT_FEATS) if USE_TEXT else 0,
                n_static_features=len(STATIC_FEATS),
                dim_feedforward=experiment_config['DISCRIMINATOR_DIM_FEEDFORWARD']
            ),
            hawkes=TransformerHawkesProcess(
                encoder=build_event_encoder(experiment_config, n_event_types),
                num_types=n_event_types
            ),
            use_text=USE_TEXT,
        )

        pretrained_state_dict = get_model_weights(model_save_dir)
        if not force_pretrain and pretrained_state_dict is not None:
            electra.load_state_dict(pretrained_state_dict, strict=False)
            print("\nPretrained model loaded successfully, skipping pretraining.\n")
            if not os.path.exists(pretrain_evaluation_fp):
                # Only reachable for weights written before this script existed, or by the
                # accelerate runner. Selecting a learning rate needs the loss those weights
                # were chosen by, and it is not recoverable after the fact.
                print(
                    f"WARNING: no pretraining evaluation at {pretrain_evaluation_fp}. The "
                    f"weights are usable but this trial cannot be ranked on pretraining loss; "
                    f"re-run with --force_pretrain if it needs to be.\n"
                )
            del pretrained_state_dict
        else:
            if force_pretrain:
                print("\nStarting pretraining from scratch.\n")
            else:
                print("\nNo pretrained model found, starting pretraining from scratch.\n")

            log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/pretrained'
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            model_save_path = f'{model_save_dir}/pretrained.pt'
            checkpoint_dir = f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/pretrained'

            timer.start_phase('pretrain', is_main_process=True)
            try:
                best_train_losses, best_val_losses = pretrain_model(
                    model=electra,
                    save_path=model_save_path,
                    loaders=dataloader_list,
                    writer=writer,
                    learning_rate=PRETRAIN_LEARNING_RATE,
                    learning_rate_decay=PRETRAIN_LEARNING_RATE_DECAY,
                    total_epoch=PRETRAIN_TOTAL_EPOCH,
                    disc_loss_weight=DISC_LOSS_WEIGHT,
                    thp_loss_nll_weight=THP_LOSS_NLL_WEIGHT,
                    thp_loss_mc_samples=THP_LOSS_MC_SAMPLES,
                    use_thp_pred_loss=USE_THP_PRED_LOSS,
                    thp_pred_loss_type_wt=THP_PRED_LOSS_TYPE_WT,
                    thp_pred_loss_time_wt=THP_PRED_LOSS_TIME_WT,
                    record_mask_ratio=RECORD_MASK_RATIO,
                    obs_unobs_sample_ratio=OBS_UNOBS_SAMPLE_RATIO,
                    cmpnt_mask_ratio=CMPNT_MASK_RATIO,
                    checkpoint_dir=checkpoint_dir,
                    accelerator=accelerator,
                    mem_test_mode=mem_test_mode,
                    ordinal_features=ordinal_features if ordinal_features else None
                )
            except Exception as e:
                print(f'Error during pretraining: {e}')
                raise
            finally:
                writer.close()
            timer.end_phase('pretrain', is_main_process=True)

            write_pretrain_evaluation(
                pretrain_evaluation_fp, EXPERIMENT_NAME, fold_name,
                best_train_losses, best_val_losses, experiment_config
            )

        # Unregister the model and optimizer from the Accelerator before the next stage
        # prepares its own. `prepare` APPENDS to an internal registry, and `save_state` writes
        # every entry in it, so a checkpoint taken during a later stage would carry this stage's
        # model as well -- and a resuming process, which reaches that stage having prepared only
        # one model, loads the wrong entry into it. The accelerate runner avoids this by
        # discarding the whole Accelerator between stages; one process needs only the unregister.
        accelerator.free_memory()
        del electra
        gc.collect()
        torch.cuda.empty_cache()

        # ---------------------------------------------------- finetuning and evaluation

        for task in task_list:

            finetuned_model_path = f'{model_save_dir}/finetuned_{task}.pt'
            evaluation_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/{task}/evaluation'
            evaluation_file = f'{evaluation_dir}/evaluation_{task}.yaml'

            if os.path.exists(evaluation_file) and not force_finetune:
                print(f"\nEvaluation for {task} already completed, skipping.\n")
                continue

            skip_finetuning = os.path.exists(finetuned_model_path) and not force_finetune
            if skip_finetuning:
                print(f"\nFinetuned {task} model found, skipping finetuning.\n")

            checkpoint_dir = f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/finetuned'
            log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/finetuned_{task}'
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            best_train_scores = None
            best_validation_scores = None

            if task == 'phenotype':
                # The phenotype listfile header determines the number of classes: the first two
                # columns are 'stay' and 'period_length', the rest are class indicators.
                phenotyping_listfile = os.path.join(fold_dir, 'phenotyping_test_listfile.csv')
                with open(phenotyping_listfile, 'r') as f_in:
                    header = f_in.readline().strip().split(',')
                    prediction_output_shape = len(header) - 2
            else:
                prediction_output_shape = 1

            def build_predictor() -> MixedClassifier:
                """Build a downstream predictor with freshly initialized encoders."""
                return MixedClassifier(
                    event_encoder=build_event_encoder(experiment_config, n_event_types),
                    val_encoder=build_value_encoder(
                        experiment_config, 'DISCRIMINATOR_ENCODER', n_val_feats, tot_val_feat_dim
                    ),
                    d_event_enc=experiment_config['THP_ENCODER_D_MODEL'],
                    d_val_enc=experiment_config['DISCRIMINATOR_ENCODER_D_MODEL'],
                    d_statics=len(STATIC_FEATS),
                    num_classes=prediction_output_shape,
                    aggr=PREDICTOR_AGGREGATION_METHOD,
                    use_text=USE_TEXT,
                )

            if not skip_finetuning:
                downstream_predictor = build_predictor()

                value_encoder_path = os.path.join(model_save_dir, 'value_encoder.pt')
                event_encoder_path = os.path.join(model_save_dir, 'event_encoder.pt')
                if not (os.path.exists(value_encoder_path) and os.path.exists(event_encoder_path)):
                    raise FileNotFoundError(
                        f"Encoder weights not found in {model_save_dir}. Expected "
                        f"value_encoder.pt and event_encoder.pt. Pretraining writes both when "
                        f"it completes, so either it has not run for this configuration or it "
                        f"did not finish."
                    )
                print(f"\nLoading encoder weights from {model_save_dir}\n")
                downstream_predictor.val_encoder.load_state_dict(
                    torch.load(value_encoder_path, map_location='cpu', weights_only=False)
                )
                downstream_predictor.event_encoder.load_state_dict(
                    torch.load(event_encoder_path, map_location='cpu', weights_only=False)
                )
                print("Successfully loaded encoder weights\n")

                timer.start_phase('finetune', is_main_process=True)
                try:
                    best_train_scores, best_validation_scores = finetune_model(
                        model=downstream_predictor,
                        save_path=finetuned_model_path,
                        loaders=[train_loader, val_loader],
                        task=task,
                        writer=writer,
                        learning_rate=FINETUNE_LEARNING_RATE,
                        learning_rate_decay=FINETUNE_LEARNING_RATE_DECAY,
                        total_epoch=FINETUNE_TOTAL_EPOCH,
                        checkpoint_dir=checkpoint_dir,
                        accelerator=accelerator,
                        mem_test_mode=mem_test_mode
                    )
                except Exception as e:
                    print(f'Error during finetuning for task {task}: {e}')
                    raise
                timer.end_phase('finetune', is_main_process=True)

                # See the note after pretraining: the evaluation below prepares a model of
                # its own, and the next task's finetune prepares another.
                accelerator.free_memory()
                del downstream_predictor
                gc.collect()
                torch.cuda.empty_cache()

            writer.close()

            # Rebuild for evaluation and load the finetuned weights from disk, so that the
            # evaluated model is the best-epoch one rather than whatever the last epoch left
            # in memory. reshape_flattened_state_dict is a near-no-op without FSDP but is kept
            # so that weights written by the accelerate runner still load here.
            downstream_predictor = build_predictor()
            finetuned_state_dict = torch.load(
                finetuned_model_path, map_location='cpu', weights_only=False
            )
            finetuned_state_dict = reshape_flattened_state_dict(
                finetuned_state_dict, get_param_shapes(downstream_predictor)
            )
            downstream_predictor.load_state_dict(finetuned_state_dict, strict=False)
            del finetuned_state_dict

            downstream_predictor = accelerator.prepare(downstream_predictor)
            best_test_scores = evaluate_finetuned_model(
                model=downstream_predictor,
                loader=test_loader,
                task=task,
                accelerator=accelerator,
                mem_test_mode=mem_test_mode
            )

            accelerator.free_memory()
            del downstream_predictor
            gc.collect()
            torch.cuda.empty_cache()

            os.makedirs(evaluation_dir, exist_ok=True)
            evaluation_data = {
                'task': task,
                'fold': fold_name,
                'experiment': EXPERIMENT_NAME,
                'hyperparameters': convert_to_python_types({
                    key: experiment_config[key]
                    for key in RECORDED_HYPERPARAMETERS if key in experiment_config
                }),
                'train_scores': convert_to_python_types(best_train_scores)
                                if best_train_scores is not None else None,
                'validation_scores': convert_to_python_types(best_validation_scores)
                                     if best_validation_scores is not None else None,
                'test_scores': convert_to_python_types(best_test_scores)
            }
            with open(evaluation_file, 'w') as f_out:
                yaml.dump(evaluation_data, f_out, default_flow_style=False, indent=2)
            print(f"Saved evaluation results to {evaluation_file}\n")

            print("\n" + format_finetuning_performance_table(
                task=task,
                train_scores=best_train_scores,
                val_scores=best_validation_scores,
                test_scores=best_test_scores
            ) + "\n")

        del dataloader_list, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

        timer.end_fold(is_main_process=True)

    timer.print_final_summary(is_main_process=True)


if __name__ == "__main__":
    main()
