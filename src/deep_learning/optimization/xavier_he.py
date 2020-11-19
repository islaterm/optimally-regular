from typing import Optional

import torch
from torch import Tensor, randn_like, reciprocal, sqrt

from deep_learning.functions.activation import ActivationFunction, relu


def he_init(weights_tensor: Tensor, randomize: bool = False) -> Tensor:
    return xavier_init(weights_tensor, randomize, 2)


def xavier_init(weights_tensor: Tensor, randomize: bool = False, factor: float = 1) -> Tensor:
    if randomize:
        weights_tensor = randn_like(weights_tensor)
    return torch.mul(weights_tensor, sqrt(reciprocal(weights_tensor.size()[1]) * factor))


activation_inits = { relu: he_init }


def xavier_he_init(weights_tensor: Tensor, randomize: bool = False,
                   activation_function: Optional[ActivationFunction] = None) -> Tensor:
    if activation_function in activation_inits:
        return activation_inits[activation_function](weights_tensor, randomize)
    return xavier_init(weights_tensor, randomize)
