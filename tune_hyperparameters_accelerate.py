"""
Accelerate-compatible version of test_tune_hyperparameters.py for multi-GPU FSDP training.

Usage:
    accelerate launch tune_hyperparameters_accelerate.py <dataset_config> <experiment_config>
"""

import argparse
import gc
import os
import torch
import yaml

from accelerate import Accelerator
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Tuple

from TransEHR2.constants import TEXT_EMBED_DIM
from TransEHR2.data.preprocessing import collate_as_tensors, prepare_dataloaders
from TransEHR2.models import ELECTRA
from TransEHR2.modules import MaskedTokenDiscriminator, MaskedTokenGenerator, TransformerHawkesProcess
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder
from TransEHR2.modules import GradientTraceableLLM
from TransEHR2.routines_accelerate import pretrain_with_hyperparameter
from TransEHR2.utils import create_timer, convert_to_python_types


class HyperparameterContext:
    """Context manager for setting hyperparameters during testing.
    
    This context manager sets one hyperparameter to a test value while setting 
    all other hyperparameters to a default value.
    """
    
    def __init__(self, hyperparameters_to_tune: list, experiment_config: dict):
        self.hyperparameters_to_tune = hyperparameters_to_tune
        self.original_values = {}
        
        # Extract values and defaults
        self.hyperparameter_values = {}
        self.hyperparameter_defaults = {}
        
        for hp_name in hyperparameters_to_tune:
            hp_list = experiment_config[hp_name]
            self.hyperparameter_values[hp_name] = hp_list.copy()
            self.hyperparameter_defaults[hp_name] = hp_list[0]
    
    def __enter__(self):
        """Enter the context - store original values."""
        for hp_name in self.hyperparameters_to_tune:
            if hp_name in globals():
                self.original_values[hp_name] = globals()[hp_name]
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context - restore original values."""
        for hp_name in self.hyperparameters_to_tune:
            if hp_name in self.original_values:
                globals()[hp_name] = self.original_values[hp_name]
            elif hp_name in globals():
                del globals()[hp_name]
        return False
    
    def set_for_testing(self, test_hp_name, test_value):
        """Set one hyperparameter for testing, others to defaults."""
        for hp_name in self.hyperparameters_to_tune:
            if hp_name == test_hp_name:
                globals()[hp_name] = test_value
            else:
                globals()[hp_name] = self.hyperparameter_defaults[hp_name]
    
    def get_values_to_test(self, hp_name):
        """Get the list of values to test for a hyperparameter."""
        return self.hyperparameter_values[hp_name].copy()
    
    def get_current_settings(self):
        """Get current hyperparameter settings."""
        return {hp: globals().get(hp, 'UNDEFINED') for hp in self.hyperparameters_to_tune}


def update_evaluation_results(
        hp_name: str, 
        value: Any, 
        best_epoch_val_losses: Dict[str, Any],
        evaluation_fp: str,
        experiment_name: str,
        fold_name: str,
        hp_context: HyperparameterContext,
        update_defaults: bool,
        accelerator: Accelerator
) -> Tuple[Dict[str, Any], bool]:
    """
    Handle loading, updating, and saving evaluation results.
    
    Only the main process performs file I/O operations.

    Args:
        hp_name (str): Name of the hyperparameter being tested.
        value (Any): Value of the hyperparameter being tested.
        best_epoch_val_losses (Dict[str, Any]): Validation losses from the best epoch using the hyperparameter setting 
            given by `hp_name` and `value`.
        evaluation_fp (str): File path to save/load evaluation results.
        experiment_name (str): Name of the experiment.
        fold_name (str): Name of the data fold.
        hp_context (HyperparameterContext): Context manager for hyperparameter settings.
        update_defaults (bool): Flag indicating whether to update results for default hyperparameter values. Typically 
            this would be done only once after the very first hyperparameter value is tested together with the default values of the rest of the hyperparameters slated for tuning. This ensures that evaluations are not repeated unnecessarily for the default value of each hyperparameter. If True, the function will set this flag to False before returning its value.
        accelerator (Accelerator): Accelerator instance for distributed training.
    
    Returns:
        tuple: A tuple containing, respectively, the updated evaluation results as a dict and the updated 
            `update_defaults` flag, which will be False if it was True when passed in.
    """
    if not accelerator.is_main_process:
        update_defaults = False
        return None, update_defaults
    
    # Load existing evaluation results or create new ones
    if os.path.exists(evaluation_fp):
        try:
            with open(evaluation_fp, 'r') as f_in:
                evaluation_results = yaml.safe_load(f_in)
        except Exception as e:
            print(f"Warning: Error loading evaluation file: {e}")
            evaluation_results = {
                'experiment': experiment_name,
                'fold': fold_name,
                'task': 'pretrain'
            }
    else:
        evaluation_results = {
            'experiment': experiment_name,
            'fold': fold_name,
            'task': 'pretrain'
        }

    # Ensure all values are Python-native types
    safe_val_losses = convert_to_python_types(best_epoch_val_losses)

    # Update evaluation results
    if hp_name not in evaluation_results:
        evaluation_results[hp_name] = {}
    if value not in evaluation_results[hp_name]:
        evaluation_results[hp_name][value] = safe_val_losses

    # Write results for all hyperparameter defaults in the first iteration
    # This avoids redundant trials with hyperparameter combos that have already been tested
    if update_defaults:
        for hp in hp_context.hyperparameters_to_tune:
            if hp not in evaluation_results:
                evaluation_results[hp] = {}
            default_value = hp_context.hyperparameter_defaults[hp]
            if default_value not in evaluation_results[hp]:
                evaluation_results[hp][default_value] = safe_val_losses.copy()
        update_defaults = False

    # Write the updated evaluation results to file
    with open(evaluation_fp, 'w') as f_out:
        yaml.dump(evaluation_results, f_out, default_flow_style=False, sort_keys=False)

    return evaluation_results, update_defaults


if __name__ == "__main__":


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tunes hyperparameters with Accelerate')
    parser.add_argument('dataset_config', type=str, help='YAML file for dataset parameters')
    parser.add_argument('experiment_config', type=str, help='YAML file for experiment parameters')
    args = vars(parser.parse_args())

    # Get dataset config parameters
    with open(args['dataset_config'], 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    TEXT_FEATS = dataset_config['TEXT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']

    # Get experiment config parameters
    with open(args['experiment_config'], 'r') as f_in:
        experiment_config = yaml.safe_load(f_in)
    EXPERIMENT_NAME = experiment_config['EXPERIMENT_NAME']
    HYPERPARAMETERS_TO_TUNE = experiment_config['HYPERPARAMETERS_TO_TUNE']
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
    DISC_LOSS_WEIGHT = experiment_config['DISC_LOSS_WEIGHT']
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


    # Create timer
    timer = create_timer(
        results_dir=f'./log/timing/{EXPERIMENT_NAME}',
        experiment_name=EXPERIMENT_NAME
    )
    timer.start_total_timing()


    # Get feature dimensions
    with open(VARIABLE_PROPERTIES_PATH, 'r') as f_in:
        variable_properties = yaml.safe_load(f_in)
    tot_val_feat_dim = 0
    numeric_feat_dims = []
    categorical_class_cnts = []
    for feature in VALUED_FEATS:
        tot_val_feat_dim += variable_properties[feature]['size']
        if variable_properties[feature]['type'] == 'numeric':
            numeric_feat_dims.append(variable_properties[feature]['size'])
        elif variable_properties[feature]['type'] == 'categorical':
            categorical_class_cnts.append(len(variable_properties[feature]['category_map']))
    if USE_TEXT:
        n_val_feats = len(VALUED_FEATS) + len(TEXT_FEATS)
        tot_val_feat_dim += len(TEXT_FEATS) * TEXT_EMBED_DIM
    else:
        n_val_feats = len(VALUED_FEATS)
    n_event_types = len(EVENT_FEATS)


    # Get fold directory
    fold_name = 'fold0'
    fold_dir = os.path.join(DATA_DIR, fold_name)


    # Setup directories
    checkpoint_dir = f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/pretrained'
    log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/pretrained'
    evaluation_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/pretrained/evaluation'
    evaluation_fp = f'{evaluation_dir}/evaluation_pretrained.yaml'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)


    # Create dataloaders
    collate_fn = partial(collate_as_tensors, device=None)  # Let Accelerator handle device placement 
    dataloader_list = prepare_dataloaders(fold_dir, BATCH_SIZE, collate_fn=collate_fn)
    train_loader, _, test_loader = dataloader_list


    with HyperparameterContext(HYPERPARAMETERS_TO_TUNE, experiment_config) as hp_context:
        
        first_iteration = True

        for hp_name in HYPERPARAMETERS_TO_TUNE:
            
            # Get the full list of HP values to test
            hp_values = hp_context.get_values_to_test(hp_name)

            # Check for existing evaluation results, exclude already tested values
            if os.path.exists(evaluation_fp):
                with open(evaluation_fp, 'r') as f_in:
                    evaluation_results = yaml.safe_load(f_in)
                if hp_name in evaluation_results:
                    tested_values = list(evaluation_results[hp_name].keys())
                    hp_values = [v for v in hp_values if v not in tested_values]
            print(f'\nTesting hyperparameter: {hp_name}')
            print(f'Values not yet tested: {hp_values}')
            

            # Iterate over hyperparameter values
            for value in hp_values:


                # Remove the accelerator to avoid issues on next iteration (will be recreated)
                # It is removed here instead of at the end of the loop to ensure that there is still an accelerator
                # instance available for writing the evalution results after the loop finishes.
                if 'accelerator' in locals():
                    del accelerator
                    torch.cuda.empty_cache()
                    gc.collect()

                # Create a fresh Accelerator instance because it solves problems.
                accelerator = Accelerator()


                locals()[hp_name] = value

                if accelerator.is_main_process:
                    print(f'\n{"="*60}')
                    print(f'TESTING: {hp_name} = {value}')
                hp_context.set_for_testing(hp_name, value)
                current_settings = hp_context.get_current_settings()
                for name, val in current_settings.items():
                    status = "TESTING" if name == hp_name else "DEFAULT"
                    if accelerator.is_main_process and status == "DEFAULT":
                        print(f'  {name} = {val} ({status})')
                if accelerator.is_main_process:
                    print(f'{"="*60}\n')
                accelerator.wait_for_everyone()


                # Initialize models
                pretrain_llm_module = GradientTraceableLLM() if USE_TEXT else None
                generator_encoder = ValueDataEncoder(
                    n_features=n_val_feats, feat_dim=tot_val_feat_dim,
                    d_model=GENERATOR_ENCODER_D_MODEL, n_heads=GENERATOR_ENCODER_N_HEADS,
                    n_encoder_blocks=GENERATOR_ENCODER_N_ENCODER_BLOCKS,
                    dim_feedforward=GENERATOR_ENCODER_DIM_FEEDFORWARD,
                    dropout=GENERATOR_ENCODER_DROPOUT,
                    activation=GENERATOR_ENCODER_ACTIVATION,
                    norm=GENERATOR_ENCODER_NORM
                )
                discriminator_encoder = ValueDataEncoder(
                    n_features=n_val_feats, feat_dim=tot_val_feat_dim,
                    d_model=DISCRIMINATOR_ENCODER_D_MODEL, n_heads=DISCRIMINATOR_ENCODER_N_HEADS,
                    n_encoder_blocks=DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
                    dim_feedforward=DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
                    dropout=DISCRIMINATOR_ENCODER_DROPOUT,
                    activation=DISCRIMINATOR_ENCODER_ACTIVATION,
                    norm=DISCRIMINATOR_ENCODER_NORM
                )
                thp_encoder = EventDataEncoder(
                    num_types=n_event_types, d_model=THP_ENCODER_D_MODEL,
                    d_inner=THP_ENCODER_D_INNER, n_layers=THP_ENCODER_N_LAYERS,
                    n_head=THP_ENCODER_N_HEADS, d_k=THP_ENCODER_D_K,
                    d_v=THP_ENCODER_D_V, dropout=THP_ENCODER_DROPOUT
                )
                generator = MaskedTokenGenerator(
                    encoder=generator_encoder, d_model=GENERATOR_D_MODEL,
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

                # Create TensorBoard writer
                if accelerator.is_main_process:
                    writer = SummaryWriter(log_dir)
                else:
                    writer = None
                accelerator.wait_for_everyone()

                # Start pretraining
                timer.start_phase('pretrain', is_main_process=accelerator.is_main_process)
                try:
                    best_epoch_train_losses, best_epoch_val_losses = pretrain_with_hyperparameter(
                        hp_name=hp_name, 
                        hp_value=value,
                        model=electra, 
                        loaders=[train_loader, test_loader],
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
                        timer=timer,
                        accelerator=accelerator
                    )
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f'Error during training with {hp_name}={value}: {e}')
                    raise
                finally:
                    if writer is not None:
                        writer.close()
                accelerator.wait_for_everyone()
                
                timer.end_phase('pretrain', is_main_process=accelerator.is_main_process)


                # Update evaluation results
                evaluation_results, first_iteration = update_evaluation_results(
                    hp_name, value, best_epoch_val_losses, evaluation_fp,
                    EXPERIMENT_NAME, fold_name, hp_context, first_iteration,
                    accelerator
                )
                accelerator.wait_for_everyone()


                # Clean up model
                del electra
                del generator
                del discriminator
                del transformer_hawkes_process
                del generator_encoder
                del discriminator_encoder
                del thp_encoder
                del pretrain_llm_module
                accelerator.wait_for_everyone()


    # Identify best hyperparameters (main process only)
    if accelerator.is_main_process:
        best_hyperparameters = {
            'experiment': EXPERIMENT_NAME,
            'fold': fold_name,
            'task': 'pretrain',
            'best_hyperparameters': {}
        }
        
        for hp_name in HYPERPARAMETERS_TO_TUNE:
            val_losses = evaluation_results[hp_name]
            best_value = None
            best_loss = float('inf')
            for value, losses in val_losses.items():
                if losses['Optimization_Loss'] < best_loss:
                    best_loss = losses['Optimization_Loss']
                    best_value = value
            best_hyperparameters['best_hyperparameters'][hp_name] = {
                'value': best_value,
                'Optimization_Loss': best_loss
            }
        
        best_hyperparameters_fp = os.path.join(evaluation_dir, 'best_hyperparameters.yaml')
        with open(best_hyperparameters_fp, 'w') as f_out:
            yaml.dump(best_hyperparameters, f_out)
        print(f'Best hyperparameters written to: {best_hyperparameters_fp}')

        timer.print_final_summary(is_main_process=accelerator.is_main_process)
