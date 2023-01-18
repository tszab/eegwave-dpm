
# Python imports
import torch
from torch import nn
from abc import ABC, abstractmethod


class DiffModelBase(nn.Module, ABC):
    def __init__(self, inp_channels: int, out_channels: int, *args, **kwargs):
        super(DiffModelBase, self).__init__()
        self._inp_channels = inp_channels
        self._out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, diff_step: torch.Tensor) -> torch.Tensor:
        pass
