"""
Contains the Base Class of the training and 2D data generating diffusion frameworks.
"""

# Python imports
import torch
from torch import nn, optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from typing import Union, Tuple, OrderedDict
from tqdm import tqdm
from copy import deepcopy

# Project imports
from diffusion_utils.diffusions import GaussianDiffusion
from ..samplers import ScheduleSampler, LossAwareSampler
from ..logging.event import TrainStatusChangedEventGroup as TrainCallbacks


class FWBase:
    """
    Unconditional diffusion framework base class designed for medical signal generation with the shape (N, C, H, W).
    """
    def __init__(
            self,
            diffusion: GaussianDiffusion,
            sampler: ScheduleSampler,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss: nn.Module,
            scheduler: optim.lr_scheduler = None,
            ema_rate: float = 0.9999,
            device: str = 'cuda',
            classes: int = None,
            grad_norm: float = 1e9,
            subjects: int = None,
            *args,
            **kwargs
    ):
        self._device = device
        self._diffusion = diffusion
        self._sampler = sampler
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._scheduler = scheduler
        self._ema_rate = ema_rate
        self._use_ema = True if ema_rate > 0.0 else False
        self._classes = classes
        self._grad_norm = grad_norm
        self._subjects = subjects

        self.publisher = TrainCallbacks()

        self._init_target_model()
        self._model = self._set_device(model)

    def _init_target_model(self):
        if self._use_ema:
            with torch.no_grad():
                self._target_model = deepcopy(self._model).cpu().state_dict()
                for tgt_param, src_param in zip(self._target_model.items(), self._model.state_dict().items()):
                    if 'weight' in tgt_param[0] or 'bias' in tgt_param[0]:
                        tgt_param[1].copy_(src_param[1].cpu())
        else:
            self._target_model = None

    def _set_device(self, to_set: Union[nn.Module, torch.Tensor]) -> Union[nn.Module, torch.Tensor]:
        to_set = to_set.cuda() if self._device != 'cpu' else to_set.cpu()
        return to_set

    def _update_target(self):
        with torch.no_grad():
            for tgt_param, src_param in zip(self._target_model.items(), self._model.state_dict().items()):
                if 'weight' in tgt_param[0] or 'bias' in tgt_param[0]:
                    tgt_param[1].mul_(self._ema_rate).add_(src_param[1].cpu(), alpha=1. - self._ema_rate)

    def _train_epoch(
            self,
            epoch: int,
            epochs: int,
            train_loader: DataLoader,
            micro_batch: int,
            eval_freq: int = 1,
            eval_samples_num: int = 10
    ) -> None:
        self._model.train()
        self.publisher.on_epoch_train_start()

        steps = len(train_loader)
        micro_batch = train_loader.batch_size if micro_batch is None else micro_batch
        test_samps_per_batch = eval_samples_num * self._classes // steps \
            if (eval_samples_num // steps > 0) and (eval_samples_num > 0) else 1
        data_shape = None

        epoch_loss = 0.
        with tqdm(total=steps, leave=True) as pbar:
            for b_idx, batch in enumerate(train_loader):
                self.publisher.on_batch_start()
                batch = list(batch)

                data = self._set_device(batch.pop(0))
                if self._classes:
                    labels = self._set_device(batch.pop(0))
                if self._subjects:
                    subjects = self._set_device(batch.pop(0))

                if data_shape is None:
                    data_shape = data.shape

                batch_loss = 0.
                self._optimizer.zero_grad()
                for i in range(0, data.shape[0], micro_batch):
                    micro_data = data[i:i + micro_batch]
                    micro_labels = labels[i:i + micro_batch] if self._classes else None
                    micro_subjects = subjects[i:i + micro_batch] if self._subjects else None

                    diff_steps, weights = self._sampler.sample(micro_data.shape[0], self._device)
                    losses = self._diffusion.training_losses(
                        self._model, micro_data, diff_steps,
                        model_kwargs={'label': micro_labels, 'subject': micro_subjects}
                    )

                    if isinstance(self._sampler, LossAwareSampler):
                        self._sampler.update_with_local_losses(diff_steps, losses["loss"].detach())

                    loss = (losses["loss"] * weights).mean()
                    loss.backward()
                    batch_loss += loss.item()

                nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_norm)
                self._optimizer.step()

                if self._scheduler is not None:
                    self._scheduler.step()

                if self._use_ema:
                    self._update_target()

                batch_loss /= data.shape[0]
                epoch_loss += batch_loss

                log_idx = torch.randperm(data.shape[0])
                self.publisher.on_batch_end({
                    'step': b_idx + epoch * steps + 1,
                    'data': data[log_idx][:test_samps_per_batch].cpu().numpy()
                    if test_samps_per_batch is not None else None,
                    'labels': labels[log_idx][:test_samps_per_batch].cpu().numpy()
                    if ((test_samps_per_batch is not None) and self._classes) else None,
                    'subjects': subjects[log_idx][:test_samps_per_batch].cpu().numpy()
                    if ((test_samps_per_batch is not None) and self._subjects) else None,
                    'loss': batch_loss
                })

                pbar.set_postfix({
                    'Epoch': '{0}/{1}'.format(epoch + 1, epochs),
                    'Loss (avg)': epoch_loss / (b_idx + 1)
                })
                pbar.update(1)

            self.publisher.on_epoch_train_end({
                'epoch': epoch + 1,
                'loss': epoch_loss / steps,
                'model': self._model,
                'target_model': self._target_model,
                'optimizer': self._optimizer,
                'scheduler': self._scheduler
            })

            if (epoch + 1) % eval_freq == 0 and eval_samples_num > 0:
                self._eval_epoch(eval_samples_num, data_shape, epoch + 1)

    def _eval_epoch(self, samples: int, shape: Tuple, epoch: int) -> None:
        classes = [label for label in range(self._classes)] if self._classes else None
        subjects = [subj for subj in range(self._subjects)] if self._subjects else None

        generated_samples, generated_labels, generated_subjects = \
            self.generate(samples, shape[1:], shape[0], True, classes, subjects)
        self.publisher.on_epoch_val_end({
            'epoch': epoch,
            'generated_data': generated_samples.numpy(),
            'generated_labels': generated_labels.numpy() if generated_labels is not None else None,
            'generated_subjects': generated_subjects.numpy() if generated_subjects is not None else None
        })

    def train(
            self,
            epochs: int,
            train_loader: DataLoader,
            micro_batch: int,
            start_epoch: int = 0,
            eval_freq: int = 1,
            eval_samples_num: int = 10,
            *args,
            **kwargs
    ) -> None:
        epochs += start_epoch
        for epoch in range(start_epoch, epochs):
            self._train_epoch(epoch, epochs, train_loader, micro_batch, eval_freq, eval_samples_num)

    @torch.no_grad()
    def generate(
            self,
            samples: int,
            shape: Tuple,
            batch_size: int = 1,
            ddim_infer: bool = True,
            classes: list = None,
            subjects: list = None,
            *args,
            **kwargs
    ) -> (torch.Tensor, None):
        self._model.eval()

        inference_fn = self._diffusion.ddim_sample_loop if ddim_infer else self._diffusion.p_sample_loop

        # Define the number of iterations
        batch_steps = samples // batch_size + (0 if (samples % batch_size) == 0 else 1)

        if classes:
            # Generate batch_step number of samples from all classes
            labels = torch.tensor(classes * batch_steps * batch_size)
            labels = one_hot(labels, num_classes=self._classes).float()
            labels = self._set_device(labels)
            batch_steps *= len(classes)
            generated_labels = []

        if subjects:
            # Generate batch_step number of samples from all classes
            subj = torch.tensor(subjects * batch_steps * batch_size)
            subj = one_hot(subj, num_classes=self._subjects).float()
            subj = self._set_device(subj)
            batch_steps *= len(subjects)
            generated_subjects = []

        generated_samples = []
        with tqdm(total=batch_steps, leave=True) as pbar:
            for batch_step in range(batch_steps):
                curr_idx = batch_step * batch_size
                batch_labels = labels[curr_idx:curr_idx + batch_size] if classes else None
                batch_subjects = subj[curr_idx:curr_idx + batch_size] if subjects else None

                data = inference_fn(
                    self._model, (batch_size, *shape),
                    model_kwargs={'label': batch_labels, 'subject': batch_subjects}
                )
                generated_samples.append(data)
                if classes:
                    generated_labels.append(batch_labels)
                if subjects:
                    generated_subjects.append(batch_subjects)
                pbar.update(1)

        generated_samples = torch.cat(generated_samples, dim=0).cpu()
        generated_labels = torch.cat(generated_labels, dim=0).cpu() if classes else None
        generated_subjects = torch.cat(generated_subjects, dim=0).cpu() if subjects else None

        # Return synthetic data in one big batch
        return generated_samples, generated_labels, generated_subjects

    def set_target_state(self, state_dict: OrderedDict):
        self._target_model = deepcopy(state_dict)
