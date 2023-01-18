"""
Weight initializing util functions
"""

# Python import
from torch import nn
from typing import Callable


def init_weights(module: nn.Module, initializer: Callable, layer_types: tuple = None, bias: bool = False) -> None:
    if layer_types is not None:
        if isinstance(module, layer_types):
            initializer(module.weight)
            if bias:
                initializer(module.bias)
    else:
        initializer(module.weight)
        if bias:
            initializer(module.bias)
