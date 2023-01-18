
# Python import
import torch
from torch import nn
from typing import Callable
from functools import partial
import numpy as np
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SASelfAttention(nn.Module):
    """
    Coursera attention implementation of the SAGAN self-attention block
    """
    def __init__(
            self,
            channels: int,
            heads: int = 1,
            channel_div: int = 8,
            conv_layer: Callable[..., nn.Module] = partial(nn.Conv2d),
            pool_layer: Callable[..., nn.Module] = partial(nn.MaxPool2d, kernel_size=(1, 2)),
            project_div: int = 1
    ) -> None:
        """
        Initializer.
        :param channels: input feature number
        :param conv_layer: convolutional layer to use, partially inited
        :param pool_layer: max pool layer to use, partially inited
        """
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.channel_div = channel_div
        self.project_div = project_div

        self.pool_phi = pool_layer()
        self.pool_g = pool_layer()

        # Spectral normalized projections
        self.theta = conv_layer(channels, channels // self.channel_div * self.heads, kernel_size=1, bias=False)
        self.phi = conv_layer(channels, channels // self.channel_div * self.heads, kernel_size=1, bias=False)
        self.g = conv_layer(channels, channels // project_div * self.heads, kernel_size=1, bias=False)
        self.o = conv_layer(channels // project_div * self.heads, channels, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2] * x.shape[-1]
        spatial_divider = 2

        # Apply convolutions to get query (theta), key (phi), and value (g) transforms
        theta = self.theta(x)
        phi = self.pool_phi(self.phi(x))
        g = self.pool_g(self.g(x))

        # Reshape spatial size for self-attention
        theta = theta.view(-1, self.channels // self.channel_div, spatial_size)
        phi = phi.view(-1, self.channels // self.channel_div, spatial_size // spatial_divider)
        g = g.view(-1, self.channels // self.project_div, spatial_size // spatial_divider)

        # Compute dot product attention with query (theta) and key (phi) matrices
        beta = torch.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)
        # Compute scaled dot product attention with value (g) and attention (beta) matrices
        o = self.o(torch.bmm(g, beta.transpose(1, 2))
                   .view(-1, self.channels // self.project_div * self.heads, *x.shape[2:]))

        # Apply gain and residual
        return self.gamma * o + x
