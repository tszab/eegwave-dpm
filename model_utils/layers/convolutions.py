"""
Specialized convolution layers
"""

# Python imports
import torch
from torch import nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from typing import Callable


class DilatedConv2d(nn.Module):
    def __init__(
            self,
            inp_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            weight_norm: Callable = None
    ):
        super(DilatedConv2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = 'same'
        self.conv = nn.Conv2d(inp_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=groups, bias=bias)
        if weight_norm is not None:
            self.conv = weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
