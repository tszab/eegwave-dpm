"""
Parameters for training, evaluation, synthesis, interpolation.
"""

# Python imports
import os
import torch
from functools import partial as p

# Project imports
from data_utils.utils import mean_norm_data, resample_tensor, extract_seq
from framework_utils.frameworks import *
from framework_utils.logging.base_logger import BasicSignalLogger
from diffusion_utils.diffusions import ContinuousDiffusion
from diffusion_utils.utils import get_named_logsnr_schedule


# Detailed configurations
FRAMEWORK_CFG = {
    'classes': 4,
    'framework': [FWBase, FWContinuous][1],
    'diffusion': ContinuousDiffusion(mean_type='x', logvar_type='fixed_large', infer_steps=1024),
    'sampler': get_named_logsnr_schedule(name='cosine', logsnr_min=-20., logsnr_max=20.),
    'model': ['eegwave'][0],
    'optimizer': ['adam', 'radam', 'adamw'][0],
    'loss': ['constant', 'snr', 'snr_trunc'][-1],
    'scheduler': [None, 'step', 'linear'][0],
    'ema_rate': 0.999,
    'ddim_infer': True,
    'grad_norm': 1.,
    'subject_embed': True
}


# Dataset details
DATASET_CFG = {
    'dataset': ['bcic4d2a', 'vepess'][1],
    'batch_size': 128,
    'micro_batch': 128,
    'train_test_ratio': 0.85,
    'val_ratio': 0.15,
    'shuffle': True,
    'shuffle_seed': 42,
    'transforms':  [
        p(extract_seq, limits=(250+125, 250+1000)),
        p(resample_tensor, down=250/128),
        mean_norm_data
        ],
    'selected_channels': None,
    'selected_subjects': None,
    'selected_classes': None,
    'split_mode': ['cross', 'within', 'mixed'][1],
    'batch_mode': [None, 'subject'][1]
}

# General configurations
RUN_CFG = {
    'mode': ['train', 'generate', 'info', 'compare_sets'][-1],
    'epochs': 5000,
    'eval_freq': 10,
    'generate_samples': 80,
    'generate_subjects': list(range(13)),
    'generate_classes': [0, 1, 2, 3],
    'generate_shape': (1, 44, 2250),
    'checkpoint_path': None,
    'ema_checkpoint_path': None,
    'generated_dir': os.path.join('dpm_generated', '1024')
}


LOGGER_CFG = {
    'logger': BasicSignalLogger,
    'log_dir': os.path.join(os.getcwd(), 'logs'),
    'save_model_epoch_freq': 10,
    'save_only_target': False,
    'sig_fs': 250.,
    'fid_model': None,
    'is_model': None,
    'top_plot_limits': (-30, 30),
    'avg_plot_limits': (-10, 10)
}


HW_PARAMS = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 1,
    'pre_fetch': 1
}
