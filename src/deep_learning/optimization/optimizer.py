"""
"optimally-regular" (c) by Ignacio Slater M.
"optimally-regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from abc import ABC, abstractmethod

import torch
from torch.nn import Parameter


class AbstractOptimizer(ABC):
    """
    Abstract class with the common elements for all optimization strategies.
    """
    _parameters: torch.nn.ParameterList
    _learning_rate: float

    def __init__(self, parameters, lr):
        self._parameters = parameters
        self._learning_rate = lr

    @abstractmethod
    def step(self):
        """Updates the parameters according to their respective gradients."""


class SGD(AbstractOptimizer):
    """
    Implementation of a Stochastic Gradient Descent optimization.
    """
    __weight_decay: float
    __momentum: float

    def __init__(self, parameters, lr, weight_decay=0, momentum=0):
        """
        Initializes a new SGD optimizer.

        Args:
            parameters:
                the parameters of the neural network to be optimized
            lr:
                the learning rate of the algorithm
            weight_decay:
                the value used for a weight decay regularization of the SGD.
            momentum:
                the momentum of the SGD optimization
        """
        super(SGD, self).__init__(parameters, lr)
        self.__weight_decay = weight_decay
        self.__momentum = momentum
        self.__momentum_buffer = None

    def step(self) -> None:
        param: Parameter
        for param in self._parameters:
            if param.grad is not None:
                d_p = param.grad
                if self.__weight_decay != 0:
                    d_p = d_p.add(param, alpha=self.__weight_decay)
                if self.__momentum != 0:
                    if self.__momentum_buffer is None:
                        self.__momentum_buffer = torch.clone(d_p).detach()
                    else:
                        self.__momentum_buffer.mul_(self.__momentum).add_(d_p)
                    d_p = self.__momentum_buffer

                torch.add(param, d_p, alpha=-self._learning_rate)
