
# Python imports
import torch
from torch import nn


class MaxNorm(nn.Module):
    def __init__(self, module: nn.Module, max_norm: float = 1.0):
        super(MaxNorm, self).__init__()
        self.module = module
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            self.module.weight.data.copy_(torch.renorm(self.module.weight.data, p=2, dim=0, maxnorm=self.max_norm))
        x = self.module(x)
        return x
