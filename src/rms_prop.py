"""
"Back to training" (c) by Ignacio Slater M.
"Back to training" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""

from typing import List

import torch
from torch import Tensor


class RMSProp:
    """Root mean square propagation implementation.
    """
    __beta: float
    __epsilon: float
    __lr: float
    __parameters: List[Tensor]
    __square_avg: Tensor
    __step: int

    def __init__(self, parameters, lr=0.001, beta=0.9, epsilon=1e-8):
        """
        Implementation of the RMSprop algorithm as proposed by G. Hinton in his
        course
        <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>.

        Args:
            parameters:
                iterable of parameters to optimize
            lr:
                the learning rate; defaults to 0.001
            beta:
                the smoothing constant; defaults to 0.9
            epsilon:
                term added to the denominator to improve numerical stability; 
                defaults to 1e-8
        """
        self.__lr = lr
        self.__beta = beta
        self.__epsilon = epsilon
        self.__parameters = parameters
        self.__step = 0
        self.__square_avg = torch.zeros_like(parameters)

    def step(self):
        """Performs a single optimization step."""
        for parameter in self.__parameters:
            if parameter.grad is not None:
                grad = parameter.grad
                self.__step += 1
                self.__square_avg = torch.addcmul(torch.mul(self.__square_avg, self.__beta), grad,
                                                  grad, value=1 - self.__beta)
                avg = torch.sqrt(self.__square_avg) + self.__epsilon
                torch.addcdiv(parameter, grad, avg, value=self.__lr, out=parameter)
