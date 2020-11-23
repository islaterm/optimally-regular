"""
"optimally-regular" (c) by Ignacio Slater M.
"optimally-regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from typing import List

import torch
from torch import Tensor

from deep_learning.optimization._base import AbstractOptimizer


class SGD(AbstractOptimizer):
    """
    Implementation of a stochastic gradient descent algorithm (optionally with momentum and weight
    decay).
    """
    __momentum: float
    __weight_decay: float
    __velocities: List[Tensor]

    def __init__(self, params: List[Tensor], lr: float, momentum: float = 0,
                 beta: float = 0):
        """
        Initializes a new SGD optimizer.

        Args:
            params:
                a list of parameters (tensors) to be optimized
            lr:
                the learning rate of the algorithm
            momentum (optional):
                the momentum used in the optimization; defaults to 0
            beta:
                the weight decay regularization factor; defaults to 0
        """
        super(SGD, self).__init__(params, lr)
        self.__weight_decay = beta
        self.__momentum = momentum
        self.__velocities = [torch.zeros_like(p) for p in self._params]

    @torch.no_grad()
    def step(self):
        i = 0
        for parameter, velocity in zip(self._params, self.__velocities):
            if parameter.grad is not None:
                grad = parameter.grad
                self._step[i] += 1

                if self.__weight_decay != 0:
                    torch.add(torch.mul(1 - self.__weight_decay, parameter),  # (1 - b) * p
                              grad, alpha=-self._learning_rate,  # grad * -lr
                              out=parameter)
                if self.__momentum != 0:
                    velocity.data = velocity.data * self.__momentum - self._learning_rate * grad
                    torch.add(parameter, velocity, out=parameter)
