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


class RMSProp(AbstractOptimizer):
    """
    Implements a root mean squared propagation algorithm as proposed by G. Hinton in his course
    <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>.
    """
    __beta: float
    __epsilon: float
    __square_avgs: List[Tensor]

    def __init__(self, params: List[Tensor], lr: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8):
        """
        Initializes a new RMSprop optimizer.

        Arguments:
            params:
                the list of parameters (tensors) to be optimized
            lr (optional):
                the learning rate; defaults to 0.001
            beta (optional):
                the smoothing constant; defaults to 0.9
            epsilon (optional):
                term added to the denominator to avoid division by 0; defaults to 1e-8
        """
        super(RMSProp, self).__init__(params, lr)
        self.__beta = beta
        self.__epsilon = epsilon
        self.__square_avgs = [torch.zeros_like(parameter) for parameter in params]

    def step(self):
        for parameter, square_avg in zip(self._params, self.__square_avgs):
            if parameter.grad is not None:
                grad = parameter.grad.data

                torch.mul(square_avg, self.__beta, out=square_avg)
                torch.addcmul(square_avg, 1 - self.__beta, grad, grad, out=square_avg)

                avg = torch.add(square_avg.sqrt(), self.__epsilon)

                parameter.data = torch.addcdiv(parameter.data, -self._learning_rate, grad, avg)
