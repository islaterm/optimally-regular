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

    @torch.no_grad()
    def step(self):
        for parameter in self._params:
            if parameter.grad is not None:
                grad = parameter.grad
                if self.__weight_decay != 0:
                    torch.add(torch.mul(1 - self.__weight_decay, parameter),  # (1 - b) * p
                              grad, alpha=-self._learning_rate,  # grad * -lr
                              out=parameter)
                if self.__momentum != 0:
                    pass
                # if self.__weight_decay != 0:
                #     d_p.add_(weight_decay, parameter.data)
                # if self.__momentum != 0:
                #     param_state = self.state[parameter]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = d_p.clone()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf
                #
                # parameter.data.add_(-group['lr'], d_p)
