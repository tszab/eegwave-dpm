
# Python imports
import os
import shutil
import numpy as np
import torch
from torch import nn, optim
from datetime import datetime

# Project imports
from ..logging.tb_base_logger import SignalTBLogger
from ..metrics.fid import FrechetInceptionDistance
from ..metrics.iscore import InceptionScore


def save_state(
        path: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        epoch: int,
        loss: float
) -> None:
    torch.save({
        'model': model.state_dict() if isinstance(model, nn.Module) else model,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'loss': loss
        }, path
    )


class BasicSignalLogger:
    def __init__(
            self,
            log_dir: str,
            save_model_epoch_freq: int = 25,
            save_only_target: bool = False,
            sig_fs: float = 250.,
            fid_model: nn.Module = None,
            is_model: nn.Module = None
    ):
        self._log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self._log_dir)
        self._tb_logger = SignalTBLogger(self._log_dir, sig_fs)

        self._save_model_freq = save_model_epoch_freq
        self._save_only_target = save_only_target
        self._fid_model = FrechetInceptionDistance(fid_model) if fid_model is not None else None
        self._is_model = InceptionScore(is_model) if is_model is not None else None
        self._batch_eval_data_list = []

        self._zip_framework()

    def _zip_framework(self):
        framework_copy_pth = os.path.join(self._log_dir, 'framework')
        os.makedirs(os.path.join(self._log_dir, 'framework'))

        for item in os.listdir(os.getcwd()):
            if item in ['logs', '.gitignore', '.idea', '.git', '__pycache__']:
                continue
            s = os.path.join(os.getcwd(), item)
            d = os.path.join(framework_copy_pth, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False)
            else:
                shutil.copy2(s, d)
        shutil.make_archive(framework_copy_pth, 'zip', framework_copy_pth)
        shutil.rmtree(framework_copy_pth)

    def reset_batch_eval_list(self):
        self._batch_eval_data_list = []

    def log_batch_end(self, log_dict: dict) -> None:
        self._tb_logger.log_loss(log_dict['step'], log_dict['loss'], 'batch')
        self._batch_eval_data_list.append([log_dict['data'], log_dict['labels'], log_dict['subjects']])

    def log_epoch_train_end(self, log_dict: dict):
        self._tb_logger.log_loss(log_dict['epoch'], log_dict['loss'], 'epoch')

        # Model state saving
        if (log_dict['epoch']) % self._save_model_freq == 0:
            # Save learning filter_models
            if log_dict['target_model'] is not None:
                save_state(os.path.join(self._log_dir, 'target_model_{}.pth'.format(log_dict['epoch'])),
                           log_dict['target_model'], log_dict['optimizer'], log_dict['scheduler'],
                           log_dict['epoch'], log_dict['loss'])
            if not self._save_only_target:
                save_state(os.path.join(self._log_dir, 'model_{}.pth'.format(log_dict['epoch'])),
                           log_dict['model'], log_dict['optimizer'], log_dict['scheduler'],
                           log_dict['epoch'], log_dict['loss'])

    def log_epoch_val_end(self, log_dict: dict):
        batch_eval_data = np.concatenate([elem[0] for elem in self._batch_eval_data_list], axis=0)
        try:
            batch_eval_labels = np.stack([elem[1] for elem in self._batch_eval_data_list], axis=0)
        except:
            batch_eval_labels = np.concatenate([elem[1] for elem in self._batch_eval_data_list], axis=0)
        batch_eval_labels, pop_idx = (None, 1) \
            if (batch_eval_labels == None).any() \
            else (batch_eval_labels.reshape(-1, batch_eval_labels.shape[-1]), 2)
        try:
            batch_eval_subjects = np.stack([elem[pop_idx] for elem in self._batch_eval_data_list], axis=0)
        except:
            batch_eval_subjects = np.concatenate([elem[pop_idx] for elem in self._batch_eval_data_list], axis=0)
        batch_eval_subjects = None \
            if (batch_eval_subjects == None).any() \
            else batch_eval_subjects.reshape(-1, batch_eval_subjects.shape[-1])

        self._tb_logger.log_pred(log_dict['epoch'],
                                 batch_eval_data.squeeze(1), log_dict['generated_data'].squeeze(1),
                                 batch_eval_labels,
                                 batch_eval_subjects,
                                 None if log_dict['generated_labels'] is None else log_dict['generated_labels'],
                                 None if log_dict['generated_subjects'] is None else log_dict['generated_subjects'],
                                 extra_tag='epoch')

        if (self._fid_model is not None) or (self._is_model is not None):
            generative_metrics = {}
            if self._fid_model is not None:
                generative_metrics.update({
                    'fid': self._fid_model(
                        log_dict['generated_data'],
                        torch.from_numpy(batch_eval_data)
                    )
                })

            if self._is_model is not None:
                generative_metrics.update({
                    'is': self._is_model(log_dict['generated_data'])
                })

            self._tb_logger.log_generative_metrics(log_dict['epoch'], generative_metrics)
