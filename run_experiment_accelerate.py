"""
Accelerate-compatible script for running experiments with multi-GPU FSDP or DDP computation.

Usage:
    accelerate launch tests/test_experiment_accelerate.py <dataset_config> <experiment_config> [--force_pretrain] [--mem_test_mode]
"""

import argparse
import gc
import os
import re
import torch
import yaml

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import broadcast, broadcast_object_list
from accelerate.utils import DistributedType
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from typing import List, Union

from TransEHR2.constants import TEXT_EMBED_DIM
from TransEHR2.data.preprocessing import collate_as_tensors, prepare_dataloaders
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import MaskedTokenDiscriminator, MaskedTokenGenerator, TransformerHawkesProcess
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder
from TransEHR2.modules import GradientTraceableLLM
from TransEHR2.routines_accelerate import pretrain_model, finetune_model, evaluate_finetuned_model
from TransEHR2.routines_accelerate import reshape_flattened_state_dict
from TransEHR2.utils import create_timer, convert_to_python_types, format_finetuning_performance_table, get_param_shapes


def initialize_accelerator(use_text: bool) -> Accelerator:
    """Initialize an Accelerate Accelerator instance with appropriate settings.
    
    If use_text is True (derived from experiment config) the model requires an LLM, which has a large memory footprint. To reduce the LLM memory usage per GPU, an accelerator is created with fully sharded data parallel (FSDP) settings. If use_text is False, the model does not require an LLM, so standard distributed data parallel (DDP, i.e. MULTI_GPU) settings suffice. Note that the accelerator config must specify the correct distrbuted computing type (FSDP or MULTI_GPU). If the configuration file does not specify the correct type, this function will raise an error.

    Args:
        use_text (bool): Whether the experiment uses text data and an LLM module.

    Returns:
        Accelerator: Configured Accelerator instance.
    
    Raises:
        ValueError: If the accelerator's distributed type does not match the expected type based on use_text.
    """

    if use_text:
        accelerator = Accelerator()
    else:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    if use_text and accelerator.distributed_type != DistributedType.FSDP:
        raise ValueError(
            f"USE_TEXT=True requires FSDP but accelerator is configured for {accelerator.distributed_type}. "
            f"Please update your accelerate config to use FSDP."
        )
    elif not use_text and accelerator.distributed_type != DistributedType.MULTI_GPU:
        raise ValueError(
            f"USE_TEXT=False expects MULTI_GPU (DDP) but accelerator is configured for {accelerator.distributed_type}. "
            f"Please update your accelerate config to use MULTI_GPU."
        )
    
    return accelerator
    

def get_fold_names(data_dir: str, exclude: List[str]):
    """Get the names of cross-validation folds from the directory tree.

    Args:
        data_dir (str): Path to the directory containing fold subdirectories
        exclude (List[str]): List of fold names to exclude from the results
    """
    fold_names = []
    for item in os.listdir(data_dir):
        if item in exclude:
            continue
        if re.match(r'fold\d+', item) and os.path.isdir(os.path.join(data_dir, item)):
            fold_names.append(item)
    fold_names.sort()
    return fold_names


def get_model_weights(dir: str, accelerator: Accelerator) -> Union[dict, None]:
    """
    Loads the state dict of the most recently trained model, if any exists in `dir`. 
    If `dir` does not exist, this function will create it.

    Args:
        dir (str): Directory to search for the optimized model weights
        accelerator (Accelerator): Accelerator instance for device management

    Returns:
        dict or None: State dict of the most recently saved optimized model, or None if no state dict found.
    """
    # Only main process checks filesystem
    if accelerator.is_main_process:
        dir_exists = os.path.exists(dir)
    else:
        dir_exists = False
    
    # Broadcast to all ranks
    dir_exists_tensor = torch.tensor(1.0 if dir_exists else 0.0, device=accelerator.device)
    dir_exists_tensor = broadcast(dir_exists_tensor, from_process=0)
    dir_exists = dir_exists_tensor.item() > 0.5
    
    if dir_exists:
        # Only main process searches for files
        if accelerator.is_main_process:
            saved_files = []
            for file in os.listdir(dir):
                if file.endswith('.pt') and os.path.isfile(os.path.join(dir, file)):
                    saved_files.append(os.path.join(dir, file))
            
            if saved_files:
                most_recent_file = max(saved_files, key=os.path.getctime)
                print(f"Loading most recently trained model's optimized weights from {most_recent_file}\n")
                state_dict = torch.load(most_recent_file, map_location='cpu', weights_only=False)
            else:
                state_dict = None
        else:
            state_dict = None
        
        # Broadcast state dict to all ranks
        state_dict = broadcast_object_list([state_dict], from_process=0)[0]
        return state_dict
    else:
        # Create directory on main process
        if accelerator.is_main_process:
            os.makedirs(dir)
        accelerator.wait_for_everyone()
        return None


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test model training and evaluation with Accelerate'
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
        '--force_pretrain', action='store_true',
        help='If specified, pretraining will be performed even if pretrained weights are found'
    )
    parser.add_argument(
        '--mem_test_mode', action='store_true',
        help='If specified, runs the pretraining forward and backward passes for a single batch to test memory usage. A message with peak memory usage will be printed before forceful termination.'
    )
    args = vars(parser.parse_args())

    force_pretrain = args['force_pretrain']
    mem_test_mode = args['mem_test_mode']

    # Get parameters from the config file(s)
    with open(args['dataset_config'], 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    TEXT_FEATS = dataset_config['TEXT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']

    with open(args['experiment_config'], 'r') as f_in:
        experiment_config = yaml.safe_load(f_in)
    EXPERIMENT_NAME = experiment_config['EXPERIMENT_NAME']
    BATCH_SIZE = experiment_config['BATCH_SIZE']
    USE_TEXT = experiment_config['USE_TEXT']
    PREDICT_INDICATORS = experiment_config['PREDICT_INDICATORS']
    GENERATOR_ENCODER_D_MODEL = experiment_config['GENERATOR_ENCODER_D_MODEL']
    GENERATOR_ENCODER_N_HEADS = experiment_config['GENERATOR_ENCODER_N_HEADS']
    GENERATOR_ENCODER_N_ENCODER_BLOCKS = experiment_config['GENERATOR_ENCODER_N_ENCODER_BLOCKS']
    GENERATOR_ENCODER_DIM_FEEDFORWARD = experiment_config['GENERATOR_ENCODER_DIM_FEEDFORWARD']
    GENERATOR_ENCODER_DROPOUT = experiment_config['GENERATOR_ENCODER_DROPOUT']
    GENERATOR_ENCODER_ACTIVATION = experiment_config['GENERATOR_ENCODER_ACTIVATION']
    GENERATOR_ENCODER_NORM = experiment_config['GENERATOR_ENCODER_NORM']
    DISCRIMINATOR_ENCODER_D_MODEL = experiment_config['DISCRIMINATOR_ENCODER_D_MODEL']
    DISCRIMINATOR_ENCODER_N_HEADS = experiment_config['DISCRIMINATOR_ENCODER_N_HEADS']
    DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS = experiment_config['DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS']
    DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD = experiment_config['DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD']
    DISCRIMINATOR_ENCODER_DROPOUT = experiment_config['DISCRIMINATOR_ENCODER_DROPOUT']
    DISCRIMINATOR_ENCODER_ACTIVATION = experiment_config['DISCRIMINATOR_ENCODER_ACTIVATION']
    DISCRIMINATOR_ENCODER_NORM = experiment_config['DISCRIMINATOR_ENCODER_NORM']
    THP_ENCODER_D_MODEL = experiment_config['THP_ENCODER_D_MODEL']
    THP_ENCODER_D_INNER = experiment_config['THP_ENCODER_D_INNER']
    THP_ENCODER_N_LAYERS = experiment_config['THP_ENCODER_N_LAYERS']
    THP_ENCODER_N_HEADS = experiment_config['THP_ENCODER_N_HEADS']
    THP_ENCODER_D_K = experiment_config['THP_ENCODER_D_K']
    THP_ENCODER_D_V = experiment_config['THP_ENCODER_D_V']
    THP_ENCODER_DROPOUT = experiment_config['THP_ENCODER_DROPOUT']
    GENERATOR_D_MODEL = experiment_config['GENERATOR_D_MODEL']
    GENERATOR_DIM_FEEDFORWARD = experiment_config['GENERATOR_DIM_FEEDFORWARD']
    DISCRIMINATOR_DIM_FEEDFORWARD = experiment_config['DISCRIMINATOR_DIM_FEEDFORWARD']
    PREDICTOR_AGGREGATION_METHOD = experiment_config['PREDICTOR_AGGREGATION_METHOD']
    MODEL_DIR = experiment_config['MODEL_DIR']
    PRETRAIN_LEARNING_RATE = experiment_config.get('PRETRAIN_LEARNING_RATE', 2e-3)
    PRETRAIN_LEARNING_RATE_DECAY = experiment_config.get('PRETRAIN_LEARNING_RATE_DECAY', 0.5)
    PRETRAIN_TOTAL_EPOCH = experiment_config.get('PRETRAIN_TOTAL_EPOCH', 2000)
    DISC_LOSS_WEIGHT = experiment_config.get('DISC_LOSS_WEIGHT', 0.5)
    THP_LOSS_NLL_WEIGHT = experiment_config.get('THP_LOSS_NLL_WEIGHT', 1e-3)
    THP_LOSS_MC_SAMPLES = experiment_config.get('THP_LOSS_MC_SAMPLES', 100)
    USE_THP_PRED_LOSS = experiment_config.get('USE_THP_PRED_LOSS', True)
    THP_PRED_LOSS_TYPE_WT = experiment_config.get('THP_PRED_LOSS_TYPE_WT', 1.0)
    THP_PRED_LOSS_TIME_WT = experiment_config.get('THP_PRED_LOSS_TIME_WT', 0.01)
    RECORD_MASK_RATIO = experiment_config.get('RECORD_MASK_RATIO', 0.25)
    OBS_UNOBS_SAMPLE_RATIO = experiment_config.get('OBS_UNOBS_SAMPLE_RATIO', 4)
    CMPNT_MASK_RATIO = experiment_config.get('CMPNT_MASK_RATIO', 0.5)
    FINETUNE_LEARNING_RATE = experiment_config.get('FINETUNE_LEARNING_RATE', 2e-4)
    FINETUNE_TOTAL_EPOCH = experiment_config.get('FINETUNE_TOTAL_EPOCH', 500)
    FINETUNE_LEARNING_RATE_DECAY = experiment_config.get('FINETUNE_LEARNING_RATE_DECAY', 0.8)

    # Create a Timer object for tracking the time it takes to train and evaluate the models
    timer = create_timer(
        results_dir=f'./log/timing/{EXPERIMENT_NAME}',
        experiment_name=EXPERIMENT_NAME
    )

    # Start timing the entire experiment
    timer.start_total_timing()

    # Get number of valued features and their sizes, class counts of categorical features, number of event types
    # These are needed as arguments for model initialization

    with open(VARIABLE_PROPERTIES_PATH, 'r') as f_in:
        variable_properties = yaml.safe_load(f_in)
    tot_val_feat_dim = 0  # Counts the total number of dimensions of all input features
    numeric_feat_dims = []  # The dimension of each numeric feature
    categorical_class_cnts = []  # The number of classes for each categorical feature
    for feature in VALUED_FEATS:
        tot_val_feat_dim += variable_properties[feature]['size'] # Accumulate numeric feature sizes
        if variable_properties[feature]['type'] == 'numeric':
            numeric_feat_dims.append(variable_properties[feature]['size'])
        elif variable_properties[feature]['type'] == 'categorical':
            categorical_class_cnts.append(len(variable_properties[feature]['category_map']))
    if USE_TEXT:
        n_val_feats = len(VALUED_FEATS) + len(TEXT_FEATS)
        tot_val_feat_dim += len(TEXT_FEATS) * TEXT_EMBED_DIM
    else:
        n_val_feats = len(VALUED_FEATS)
    # Get number of event types
    n_event_types = len(EVENT_FEATS)


    # Get cross validation fold subdirectories' names and iterate over folds
    fold_name_list = get_fold_names(DATA_DIR, exclude='fold0')  # Fold0 is reserved for hyperparameter tuning

    for fold_name in fold_name_list:
        # Create a fresh accelerator because it solves problems
        if 'accelerator' in locals():
            # free_memory clears internal refs, runs gc.collect, and empties cuda cache
            accelerator.free_memory()
            del accelerator
            # Optional: one more round of garbage collection to be safe
            gc.collect()
            torch.cuda.empty_cache()
        accelerator = initialize_accelerator(USE_TEXT)


        # Start timing this fold
        timer.start_fold(fold_name)


        fold_dir = os.path.join(DATA_DIR, fold_name)
        # Create the list of dataloaders for the training, validation (optional), and test sets
        # Dataloaders serve batched MixedDataset objects
        collate_fn = partial(collate_as_tensors, device=None)  # Accelerate will handle device placement
        dataloader_list = prepare_dataloaders(fold_dir, BATCH_SIZE, collate_fn=collate_fn)
        train_loader, val_loader, test_loader = dataloader_list


        # Initialize models
        if USE_TEXT:
            pretrain_llm_module = GradientTraceableLLM()
        else:
            pretrain_llm_module = None
        generator_encoder = ValueDataEncoder(
            n_features=n_val_feats,
            feat_dim=tot_val_feat_dim,
            d_model=GENERATOR_ENCODER_D_MODEL,
            n_heads=GENERATOR_ENCODER_N_HEADS,
            n_encoder_blocks=GENERATOR_ENCODER_N_ENCODER_BLOCKS,
            dim_feedforward=GENERATOR_ENCODER_DIM_FEEDFORWARD,
            dropout=GENERATOR_ENCODER_DROPOUT,
            activation=GENERATOR_ENCODER_ACTIVATION,
            norm=GENERATOR_ENCODER_NORM
        )
        discriminator_encoder = ValueDataEncoder(
            n_features=n_val_feats,
            feat_dim=tot_val_feat_dim,
            d_model=DISCRIMINATOR_ENCODER_D_MODEL,
            n_heads=DISCRIMINATOR_ENCODER_N_HEADS,
            n_encoder_blocks=DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
            dim_feedforward=DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
            dropout=DISCRIMINATOR_ENCODER_DROPOUT,
            activation=DISCRIMINATOR_ENCODER_ACTIVATION,
            norm=DISCRIMINATOR_ENCODER_NORM
        )
        thp_encoder = EventDataEncoder(
            num_types=n_event_types,
            d_model=THP_ENCODER_D_MODEL,
            d_inner=THP_ENCODER_D_INNER,
            n_layers=THP_ENCODER_N_LAYERS,
            n_head=THP_ENCODER_N_HEADS,
            d_k=THP_ENCODER_D_K,
            d_v=THP_ENCODER_D_V,
            dropout=THP_ENCODER_DROPOUT
        )
        generator = MaskedTokenGenerator(
            encoder=generator_encoder,
            d_model=GENERATOR_D_MODEL,
            numeric_dims=numeric_feat_dims,
            categorical_classes=categorical_class_cnts,
            n_text_features=len(TEXT_FEATS) if USE_TEXT else 0,
            predict_indicators=PREDICT_INDICATORS,
            dim_feedforward=GENERATOR_DIM_FEEDFORWARD
        )
        discriminator = MaskedTokenDiscriminator(
            encoder=discriminator_encoder,
            d_model=DISCRIMINATOR_ENCODER_D_MODEL,
            n_numeric_features=len(numeric_feat_dims),
            n_categorical_features=len(categorical_class_cnts),
            n_text_features=len(TEXT_FEATS) if USE_TEXT else 0,
            n_static_features=len(STATIC_FEATS),
            dim_feedforward=DISCRIMINATOR_DIM_FEEDFORWARD
        )
        transformer_hawkes_process = TransformerHawkesProcess(
            encoder=thp_encoder,
            num_types=n_event_types
        )
        electra = ELECTRA(
            generator=generator,
            discriminator=discriminator,
            hawkes=transformer_hawkes_process,
            use_text=USE_TEXT,
            llm_module=pretrain_llm_module  # Will be None when USE_TEXT is False
        )


        # If we're not forcing pretraining, load the most recently saved pretrained model if one exists
        model_save_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/pretrained'
        pretrained_state_dict = get_model_weights(model_save_dir, accelerator)
        if not force_pretrain and pretrained_state_dict is not None:
            electra.load_state_dict(pretrained_state_dict, strict=False)
            if accelerator.is_main_process:
                print("\nPretrained model loaded successfully, skipping pretraining.\n")
            del pretrained_state_dict
        else:
            if accelerator.is_main_process:
                if force_pretrain:
                    print("\nStarting pretraining from scratch.\n")
                else:
                    print("\nNo pretrained model found, starting pretraining from scratch.\n")

            # Timestamp for file names; marks the beginning of pretraining
            log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/pretrained'
            if accelerator.is_main_process:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir)
            else:
                writer = None
            accelerator.wait_for_everyone()
            

            # Paths for saving model and checkpoints
            model_save_path = f'{model_save_dir}/pretrained.pt'
            checkpoint_dir = f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/pretrained'


            # Get parameter shapes before the model is wrapped by Accelerator
            if accelerator.is_main_process:
                pretrain_param_shapes = get_param_shapes(electra)
            else:
                pretrain_param_shapes = None
            pretrain_param_shapes = broadcast_object_list([pretrain_param_shapes], from_process=0)[0]


            # Pretrain the model
            timer.start_phase('pretrain', is_main_process=accelerator.is_main_process)
            try:
                best_train_losses, best_val_losses = pretrain_model(
                    model=electra,
                    save_path=model_save_path,
                    loaders=dataloader_list,  # Training set dataloader
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
                    mem_test_mode=mem_test_mode
                )
            except Exception as e:
                if accelerator.is_main_process:
                    print(f'Error during pretraining: {e}')
                raise
            timer.end_phase('pretrain', is_main_process=accelerator.is_main_process)

            # Clean up
            if writer is not None:
                writer.close()
                del writer

        # Clean up
        del electra
        del generator
        del discriminator
        del transformer_hawkes_process
        del generator_encoder
        del discriminator_encoder
        del thp_encoder
        del pretrain_llm_module
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


        # Supervised task-specific finetuning and evaluation
        for task in ['mortality', 'length_of_stay', 'phenotype']:

            # Create a fresh accelerator because it solves problems
            if 'accelerator' in locals():
                # free_memory clears internal refs, runs gc.collect, and empties cuda cache
                accelerator.free_memory()
                del accelerator
                # Optional: one more round of garbage collection to be safe
                gc.collect()
                torch.cuda.empty_cache()
            accelerator = initialize_accelerator(USE_TEXT)

            checkpoint_dir = f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/finetuned'
            log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/finetuned_{task}'
            if accelerator.is_main_process:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir)
            else:
                writer = None
            accelerator.wait_for_everyone()
            

            # Reset the best scores for each new task
            best_train_scores = None
            best_validation_scores = None
            best_test_scores = None


            if task == 'phenotype':
                # Load a phenotype listfile to determine how many classes there are
                phenotyping_listfile = os.path.join(fold_dir, 'phenotyping_test_listfile.csv')
                with open(phenotyping_listfile, 'r') as f_in:
                    header = f_in.readline().strip().split(',')
                    # The first two columns are 'stay' and 'period_length', the rest are phenotype class indicators
                    prediction_output_shape = len(header) - 2
            else:
                prediction_output_shape = 1


            # Create a new LLM module for this task
            if USE_TEXT:
                finetune_llm_module = GradientTraceableLLM()
            else:
                finetune_llm_module = None
            # Initialize the downstream predictor. The encoders are reinitialized with pretraining parameters.
            predictor_value_encoder = ValueDataEncoder(
                n_features=n_val_feats,
                feat_dim=tot_val_feat_dim,
                d_model=DISCRIMINATOR_ENCODER_D_MODEL,
                n_heads=DISCRIMINATOR_ENCODER_N_HEADS,
                n_encoder_blocks=DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
                dim_feedforward=DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
                dropout=DISCRIMINATOR_ENCODER_DROPOUT,
                activation=DISCRIMINATOR_ENCODER_ACTIVATION,
                norm=DISCRIMINATOR_ENCODER_NORM
            )
            predictor_event_encoder = EventDataEncoder(
                num_types=n_event_types,
                d_model=THP_ENCODER_D_MODEL,
                d_inner=THP_ENCODER_D_INNER,
                n_layers=THP_ENCODER_N_LAYERS,
                n_head=THP_ENCODER_N_HEADS,
                d_k=THP_ENCODER_D_K,
                d_v=THP_ENCODER_D_V,
                dropout=THP_ENCODER_DROPOUT
            )
            downstream_predictor = MixedClassifier(
                event_encoder=predictor_event_encoder,
                val_encoder=predictor_value_encoder,
                d_event_enc=THP_ENCODER_D_MODEL,
                d_val_enc=DISCRIMINATOR_ENCODER_D_MODEL,
                d_statics=len(STATIC_FEATS),
                num_classes=prediction_output_shape,
                aggr=PREDICTOR_AGGREGATION_METHOD,
                use_text=USE_TEXT,
                llm_module=finetune_llm_module  # Will be None when USE_TEXT is False
            )

            # Load pretrained encoder weights into downstream predictor
            pretrained_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/pretrained'
            value_encoder_path = os.path.join(pretrained_dir, 'value_encoder.pt')
            event_encoder_path = os.path.join(pretrained_dir, 'event_encoder.pt')
            if accelerator.is_main_process:
                encoders_exist = os.path.exists(value_encoder_path) and os.path.exists(event_encoder_path)
                if not encoders_exist:
                    raise FileNotFoundError(
                        f"Encoder weights not found in {pretrained_dir}. "
                        f"Expected files: value_encoder.pt and event_encoder.pt. "
                        f"Make sure pretraining completed successfully."
                    )
                else:
                    print(f"\nLoading encoder weights from {pretrained_dir}\n")
                    value_encoder_state = torch.load(value_encoder_path, map_location='cpu', weights_only=False)
                    event_encoder_state = torch.load(event_encoder_path, map_location='cpu', weights_only=False)
            else:
                value_encoder_state = None
                event_encoder_state = None
            value_encoder_state = broadcast_object_list([value_encoder_state], from_process=0)[0]
            event_encoder_state = broadcast_object_list([event_encoder_state], from_process=0)[0]
            downstream_predictor.val_encoder.load_state_dict(value_encoder_state)
            downstream_predictor.event_encoder.load_state_dict(event_encoder_state)
            del value_encoder_state
            del event_encoder_state
            if accelerator.is_main_process:
                    print("Successfully loaded encoder weights\n")
            accelerator.wait_for_everyone()


            # Get parameter shapes before wrapping with Accelerator
            if accelerator.is_main_process:
                finetune_param_shapes = get_param_shapes(downstream_predictor)
            else:
                finetune_param_shapes = None
            finetune_param_shapes = broadcast_object_list([finetune_param_shapes], from_process=0)[0]


            # Supervised task-specific finetuning
            # Returns the scores from the best-performing epoch as determined by validation set performance
            timer.start_phase('finetune', is_main_process=accelerator.is_main_process)
            try:
                best_train_scores, best_validation_scores = finetune_model(
                    model=downstream_predictor,
                    save_path=f'{model_save_dir}/finetuned_{task}.pt',
                    loaders=[train_loader, val_loader],
                    task=task,
                    writer=writer,
                    learning_rate=FINETUNE_LEARNING_RATE,
                    learning_rate_decay=FINETUNE_LEARNING_RATE_DECAY,
                    total_epoch=FINETUNE_TOTAL_EPOCH,
                    checkpoint_dir=checkpoint_dir,
                    accelerator=accelerator
                )
            except Exception as e:
                if accelerator.is_main_process:
                    print(f'Error during finetuning for task {task}: {e}')
                raise
            timer.end_phase('finetune', is_main_process=accelerator.is_main_process)


            # Clean up
            if writer is not None:
                writer.close()
                del writer
            del downstream_predictor
            del predictor_value_encoder
            del predictor_event_encoder
            del finetune_llm_module
            del accelerator
            gc.collect()
            torch.cuda.empty_cache()


            # Create a fresh accelerator because it solves problems
            accelerator = initialize_accelerator(USE_TEXT)

            # Reinitialize the model for evaluation
            if USE_TEXT:
                finetune_llm_module = GradientTraceableLLM()
            else:
                finetune_llm_module = None
            predictor_value_encoder = ValueDataEncoder(
                n_features=n_val_feats,
                feat_dim=tot_val_feat_dim,
                d_model=DISCRIMINATOR_ENCODER_D_MODEL,
                n_heads=DISCRIMINATOR_ENCODER_N_HEADS,
                n_encoder_blocks=DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
                dim_feedforward=DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
                dropout=DISCRIMINATOR_ENCODER_DROPOUT,
                activation=DISCRIMINATOR_ENCODER_ACTIVATION,
                norm=DISCRIMINATOR_ENCODER_NORM
            )
            predictor_event_encoder = EventDataEncoder(
                num_types=n_event_types,
                d_model=THP_ENCODER_D_MODEL,
                d_inner=THP_ENCODER_D_INNER,
                n_layers=THP_ENCODER_N_LAYERS,
                n_head=THP_ENCODER_N_HEADS,
                d_k=THP_ENCODER_D_K,
                d_v=THP_ENCODER_D_V,
                dropout=THP_ENCODER_DROPOUT
            )
            downstream_predictor = MixedClassifier(
                event_encoder=predictor_event_encoder,
                val_encoder=predictor_value_encoder,
                d_event_enc=THP_ENCODER_D_MODEL,
                d_val_enc=DISCRIMINATOR_ENCODER_D_MODEL,
                d_statics=len(STATIC_FEATS),
                num_classes=prediction_output_shape,
                aggr=PREDICTOR_AGGREGATION_METHOD,
                use_text=USE_TEXT,
                llm_module=finetune_llm_module
            )

            # Load the finetuned state dict from file on the main process
            finetuned_state_dict_path = f'{model_save_dir}/finetuned_{task}.pt'
            if accelerator.is_main_process:
                finetuned_state_dict = torch.load(finetuned_state_dict_path, map_location='cpu', weights_only=False)
                # Reshape it using the unwrapped model as reference
                finetuned_state_dict = reshape_flattened_state_dict(finetuned_state_dict, finetune_param_shapes)
            else:
                finetuned_state_dict = None
            # Broadcast the reshaped finetuned state dict to all ranks
            finetuned_state_dict = broadcast_object_list([finetuned_state_dict], from_process=0)[0]
            # Load the finetuned weights into the downstream predictor on all processes
            downstream_predictor.load_state_dict(finetuned_state_dict, strict=False)
            # Free memory
            del finetuned_state_dict
            accelerator.wait_for_everyone()


            # Prepare the model and dataloader with Accelerator and evaluate on test set
            downstream_predictor = accelerator.prepare(downstream_predictor)
            accelerator.wait_for_everyone()
            wrapped_test_loader = accelerator.prepare(test_loader)  # Original test_loader stays unwrapped
            best_test_scores = evaluate_finetuned_model(
                model=downstream_predictor,
                loader=wrapped_test_loader,
                task=task,
                accelerator=accelerator
            )


            # Clean up model
            del downstream_predictor
            del predictor_value_encoder
            del predictor_event_encoder
            del finetune_llm_module
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()


            # Save and print results from this task
            if accelerator.is_main_process:
                best_train_scores = convert_to_python_types(best_train_scores)
                best_validation_scores = convert_to_python_types(best_validation_scores)
                best_test_scores = convert_to_python_types(best_test_scores)
                # Save performance metrics from the best models to YAML files
                evaluation_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/{task}/evaluation'
                os.makedirs(evaluation_dir, exist_ok=True)
                evaluation_data = {
                    'task': task,
                    'fold': fold_name,
                    'experiment': EXPERIMENT_NAME,
                    'train_scores': best_train_scores,
                    'validation_scores': best_validation_scores,
                    'test_scores': best_test_scores
                }
                evaluation_file = f'{evaluation_dir}/evaluation_{task}.yaml'
                with open(evaluation_file, 'w') as f_out:
                    yaml.dump(evaluation_data, f_out, default_flow_style=False, indent=2)
                print(f"Saved evaluation results to {evaluation_file}\n")
                # Print summary of results
                performance_table = format_finetuning_performance_table(
                    task=task,
                    train_scores=best_train_scores,
                    val_scores=best_validation_scores,
                    test_scores=best_test_scores
                )
                print("\n" + performance_table + "\n")
    

            accelerator.wait_for_everyone()

        # Clean up
        del dataloader_list
        del train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


        timer.end_fold(is_main_process=accelerator.is_main_process)


    if accelerator.is_main_process:
        timer.print_final_summary(is_main_process=accelerator.is_main_process)