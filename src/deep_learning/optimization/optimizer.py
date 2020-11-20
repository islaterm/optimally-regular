"""
"optimally-regular" (c) by Ignacio Slater M.
"optimally-regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor
from torch.nn import Parameter

from autocorrect import corrector, token


class AbstractOptimizer(ABC):
    """
    Abstract class with the common elements for all optimization strategies.
    """
    _parameters: List[Tensor]
    _learning_rate: float

    def __init__(self, parameters, lr):
        self._parameters = [p for p in parameters if p is not None]
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


class RMSProp(AbstractOptimizer):
    """Root mean square propagation implementation.
    """
    __beta: float
    __epsilon: float
    _learning_rate: float
    __square_avg: List[Tensor]

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
        super(RMSProp, self).__init__(parameters, lr)
        self.__beta = beta
        self.__epsilon = epsilon
        self.__square_avg = [torch.zeros_like(w) for w in self._parameters]

    def step(self):
        for parameter, avg in zip(self._parameters, self.__square_avg):
            if parameter.grad is not None:
                avg.data = torch.mul(self.__beta, avg) + torch.mul((1 - self.__beta),
                                                                   torch.square(parameter.grad))
                parameter.data = parameter.data - self._learning_rate * (
                        1 / torch.sqrt(avg.data) + self.__epsilon) * parameter.grad


if __name__ == '__main__':
    # Tests del API del curso
    weight, grad = corrector.get_test_data(homework=3, question="2c", test=1, token=token)

    weight = torch.tensor(weight, requires_grad=True)
    weight.grad = torch.tensor(grad)

    optimizer = RMSProp([weight], lr=0.001, beta=0.9, epsilon=1e-8)
    optimizer.step()

    # Submit
    corrector.submit(homework=3, question="2c", test=1, token=token, answer=weight)
    optimizer.step()
    corrector.submit(homework=3, question="2c", test=2, token=token, answer=weight)
