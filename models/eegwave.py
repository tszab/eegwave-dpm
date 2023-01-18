"""
Implementation of the architecture
"""

# Python imports
import numpy as np
from typing import Tuple, Union
import torch
from torch import nn
from typing import Callable

# Project imports
from .diff_model_base import DiffModelBase
from model_utils.layers import EEGWaveResidual, LogSNREmbedding


def inited_module(module: nn.Module, init: Callable):
    init(module.weight)
    return module


class EEGWaveTempNet(nn.Module):
    def __init__(
            self,
            inp_channels: int,
            out_channels: int,
            res_channels: int,
            skip_channels: int,
            res_layers: int = 30,
            dilation_cycles: int = 10,
            step_embed_dim: int = 512,
            kernel_size: Tuple = (3, 3)
    ):
        super().__init__()
        self.out_norm_value = np.sqrt(res_layers)

        self.inp_seq = nn.Sequential(
            inited_module(
                nn.Conv2d(inp_channels, res_channels, kernel_size=(kernel_size[0], 1)),
                nn.init.kaiming_normal_),
            nn.SiLU()
        )

        # Residual blocks
        self.res_layers = nn.ModuleList([
            EEGWaveResidual(res_channels, skip_channels, step_embed_dim, False,
                            kernel_size=kernel_size[1], dilation=2**(i % dilation_cycles),
                            norm_value=np.sqrt(2.))
            for i in range(res_layers)
        ])

        # Output convolution
        self.out_seq = nn.Sequential(
            inited_module(
                nn.Conv2d(skip_channels, skip_channels, kernel_size=(1, 1)),
                nn.init.kaiming_normal_
            ),
            nn.SiLU(),
            inited_module(
                nn.Conv2d(skip_channels, out_channels, kernel_size=(1, 1)),
                nn.init.zeros_
            ),
        )

    def forward(self, x: torch.Tensor, diff_step: torch.Tensor) -> torch.Tensor:
        x = self.inp_seq(x)
        skip, res = self.res_layers[0](x, diff_step)
        if len(self.res_layers) > 1:
            for res_layer in self.res_layers[1:]:
                skip_add, res = res_layer(res, diff_step)
                skip = skip + skip_add
        skip = skip / self.out_norm_value
        out_data = self.out_seq(skip)
        return out_data


class EEGWave(DiffModelBase):
    def __init__(
            self,
            eeg_channels: int,
            inp_channels: int,
            out_channels: int,
            res_channels: int = 256,
            skip_channels: int = 256,
            res_layers: int = 30,
            dilation_cycles: int = 10,
            step_embed_dim: int = 512,
            kernel_size: int = 3,
            classes: int = None,
            subjects: int = None,
            interpolate: bool = False,
            *args,
            **kwargs
    ):
        super(EEGWave, self).__init__(1, 1)
        # Step embedding
        self.step_embed_dim = step_embed_dim
        self.eeg_channels = eeg_channels
        self.out_channels = out_channels
        self.split = self.out_channels > self.eeg_channels
        self.step_embedding = LogSNREmbedding('inv_cos', 128, step_embed_dim)

        self.temporal_net = EEGWaveTempNet(inp_channels, out_channels, res_channels, skip_channels,
                                           res_layers, dilation_cycles, step_embed_dim, (eeg_channels, kernel_size))

        if classes is not None:
            self.label_embed = nn.Linear(classes, step_embed_dim)
        if subjects is not None:
            self.subj_embed = nn.Linear(subjects, step_embed_dim)

        self.interpolate = interpolate

    def _interpolate(self, subjects: torch.Tensor):
        each_subject_embed = torch.eye(subjects.shape[-1], device=subjects.device, dtype=subjects.dtype)
        each_subject_embed = self.subj_embed(each_subject_embed)
        return torch.matmul(subjects, each_subject_embed)

    def forward(
            self,
            x: torch.Tensor,
            diff_step: torch.Tensor = torch.tensor(0.).view(1, 1),
            label: torch.Tensor = None,
            subject: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff_step = self.step_embedding(diff_step)
        if label is not None:
            label = self.label_embed(label)
            diff_step = diff_step + label
        if subject is not None:
            if self.interpolate:
                subject = self._interpolate(subject)
            else:
                subject = self.subj_embed(subject)
            diff_step = diff_step + subject

        out = self.temporal_net(x, diff_step)
        if self.split:
            out = torch.cat(torch.chunk(out, self.out_channels//self.eeg_channels, dim=1), dim=2)
        out = out.transpose(1, 2).contiguous()

        return out
