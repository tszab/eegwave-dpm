"""
Loss configuration dictionaries
"""

import torch
from torch import nn
from functools import partial

from model_utils.losses import LogL1Loss


LOSSES = {
    'mse': partial(nn.MSELoss, reduction='mean'),
    'l1': partial(nn.L1Loss, reduction='mean'),
    'log_l1': LogL1Loss
}
