"""
Optimizer configuration dictionaries
"""

from torch import optim
from functools import partial


OPTIMIZERS = {
    'adam': partial(optim.Adam, lr=1e-4, betas=(0.9, 0.999), eps=1e-9, weight_decay=0., amsgrad=False),
    'radam': partial(optim.RAdam, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.),
    'adamw': partial(optim.AdamW, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
}
