"""
Residual layer implementations
"""

# Python imports
import math
import torch
from torch import nn
from functools import partial

# Project imports
from .convolutions import DilatedConv2d
from ..weight_utils.initializers import init_weights
from .scalers import Upsample, Downsample


class EEGWaveResidual(nn.Module):
    def __init__(
            self,
            res_channels: int,
            skip_channels: int,
            step_embed_dim: int = 512,
            multi_channel: bool = False,
            condition_dim: int = None,
            kernel_size: int = 3,
            dilation: int = 1,
            norm_value: float = math.sqrt(2.)
    ):
        super(EEGWaveResidual, self).__init__()
        assert norm_value > 0.

        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.multi_channel = multi_channel
        self.conditioned = True if (condition_dim is not None) and (condition_dim > 0) else False
        self.register_buffer('norm_value', torch.tensor(norm_value))

        # Layer specific step embedding
        self.step_embedding = nn.Linear(step_embed_dim, self.res_channels)

        # Dilated separated conv
        self.temporal_conv = DilatedConv2d(self.res_channels, 2 * self.res_channels, kernel_size=(1, kernel_size),
                                           dilation=(1, dilation))#, weight_norm=nn.utils.weight_norm)
        self.temporal_conv.apply(partial(init_weights, initializer=nn.init.kaiming_normal_,
                                         layer_types=nn.Conv2d, bias=False))

        self.skip_res_conv = nn.Conv2d(self.res_channels, self.skip_channels + self.res_channels,
                                       kernel_size=(1, 1))
        nn.init.kaiming_normal_(self.skip_res_conv.weight)

    def forward(self, x: torch.Tensor, diff_step: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        diff_step = self.step_embedding(diff_step)
        diff_step = diff_step.view(*diff_step.shape, 1, 1).contiguous()

        h = x + diff_step
        h = self.temporal_conv(h)

        gate, filt = torch.chunk(h, 2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filt)
        h = self.skip_res_conv(h)

        skip, res = torch.split(h, [self.skip_channels, self.res_channels], dim=1)
        res = (x + res) / self.norm_value
        return skip, res


class ResBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        scale_conv=False,
        kernel_size=3
    ):
        super().__init__()
        self.inp_channels = inp_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or inp_channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        gn_param = 8
        pad = kernel_size//2

        self.in_layers = nn.Sequential(
            nn.GroupNorm(gn_param, self.inp_channels),
            nn.SiLU(),
            nn.Conv2d(self.inp_channels, self.out_channels, (kernel_size, 1), padding=(pad, 0)),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.inp_channels, scale_conv, factor=(2, 1))
            self.x_upd = Upsample(self.inp_channels, scale_conv, factor=(2, 1))
        elif down:
            self.h_upd = Downsample(self.inp_channels, scale_conv, factor=(2, 1))
            self.x_upd = Downsample(self.inp_channels, scale_conv, factor=(2, 1))
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(gn_param, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, (kernel_size, 1), padding=(pad, 0))
        )
        for p in self.out_layers[-1].parameters():
            p.detach().zero_()

        if self.out_channels == self.inp_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(self.inp_channels, self.out_channels, (kernel_size, 1), padding=(pad, 0))
        else:
            self.skip_connection = nn.Conv2d(self.inp_channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
