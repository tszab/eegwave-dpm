"""
Optimizer configuration dictionaries
"""

from torch.optim import lr_scheduler
from functools import partial


SCHEDULERS = {
    'step': partial(lr_scheduler.StepLR, step_size=100, gamma=0.5),
    'linear': partial(lr_scheduler.LinearLR, start_factor=1.0, end_factor=0.0, total_iters=10)
}

