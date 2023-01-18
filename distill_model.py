"""
Main file for distillating a trained diffusion model.
"""

# Python imports
import numpy as np
from copy import deepcopy

# Project imports
from configurations import *
from run_config import *
from torch.utils.data import DataLoader, Dataset

# Main parameters
DISTILL_EPOCHS = 22
START_INFER_STEPS = 1024
END_INFER_STEPS = 1
SAVE_PATH = r''

if __name__ == '__main__':
    def init_dataloader(dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, DATASET_CFG['batch_size'],
                          DATASET_CFG['shuffle'], num_workers=HW_PARAMS['num_workers'],
                          prefetch_factor=HW_PARAMS['pre_fetch'])


    # Initialize dataset and dataloaders
    dc = DATASET_CFG
    dataset = DATASETS[DATASET_CFG['dataset']](**dc)
    if (dc['val_ratio'] is not None) and (dc['val_ratio'] > 0.0):
        train_set, eval_set, _ = dataset.split_set(dc['train_test_ratio'], dc['split_mode'], dc['shuffle'],
                                                   dc['shuffle_seed'], None,
                                                   val_split=dc['val_ratio'])
    else:
        train_set, eval_set = dataset.split_set(dc['train_test_ratio'], dc['split_mode'], dc['shuffle'],
                                                dc['shuffle_seed'], None,
                                                val_split=dc['val_ratio'])

    subjects = np.unique([train_set.dataset.subjects, eval_set.dataset.subjects])
    train_loader = init_dataloader(train_set)
    eval_loader = init_dataloader(eval_set)

    # Initialize framework and models
    fc = FRAMEWORK_CFG
    student = MODELS[fc['model']](classes=fc['classes'], subjects=len(subjects) if fc['subject_embed'] else None)
    teacher = MODELS[fc['model']](classes=fc['classes'], subjects=len(subjects) if fc['subject_embed'] else None)
    optimizer = OPTIMIZERS[fc['optimizer']]
    scheduler = SCHEDULERS[fc['scheduler']]

    checkpoint = RUN_CFG['checkpoint_path']
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        teacher.load_state_dict(deepcopy(checkpoint['model']))
        student.load_state_dict(deepcopy(checkpoint['model']))
    else:
        raise AttributeError('A teacher checkpoint must have to be added!')

    diffusion = fc['diffusion']
    sampler = fc['sampler']
    framework = Distillation(diffusion, sampler, student, optimizer, fc['loss'], scheduler, fc['ema_rate'],
                             HW_PARAMS['device'], fc['classes'], fc['grad_norm'],
                             subjects=len(subjects) if fc['subject_embed'] else None)

    # Initialize logger
    logger = LOGGER_CFG['logger'](log_dir=LOGGER_CFG['log_dir'], save_only_target=LOGGER_CFG['save_only_target'],
                                  save_model_epoch_freq=LOGGER_CFG['save_model_epoch_freq'],
                                  sig_fs=LOGGER_CFG['sig_fs'], fid_model=LOGGER_CFG['fid_model'],
                                  is_model=LOGGER_CFG['is_model'])

    # Subscribe to training events with the logger
    framework.publisher.on_batch_end.append(lambda x: logger.log_batch_end(x))
    framework.publisher.on_epoch_train_end.append(lambda x: logger.log_epoch_train_end(x))
    framework.publisher.on_epoch_val_end.append(lambda x: logger.log_epoch_val_end(x))
    framework.publisher.on_epoch_val_end.append(lambda _: logger.reset_batch_eval_list())

    framework.train(teacher, DISTILL_EPOCHS, train_loader, DATASET_CFG['micro_batch'],
                    RUN_CFG['eval_freq'], RUN_CFG['generate_samples'], START_INFER_STEPS, END_INFER_STEPS)

    if SAVE_PATH:
        torch.save(
            {'model': student.state_dict()},
            os.path.join(os.getcwd(), 'model_end.pth')
        )

