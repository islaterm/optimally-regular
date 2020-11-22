import math
from typing import List, Tuple

import torch
from torch import Tensor

from deep_learning.optimization._base import AbstractOptimizer


class Adam(AbstractOptimizer):
    """
    Implements Adam optimization algorithm as proposed in "Adam: A Method for Stochastic
    Optimization".
    """
    __betas: Tuple[float, float]
    __epsilon: float
    __exp_avgs: List[Tensor]
    __squared_exp_avgs: List[Tensor]

    def __init__(self, params: List[Tensor], lr: float = 1e-3, beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        """
        Initializes a new Adam optimizer.

        Arguments:
            params (iterable):
                the list of parameters (tensors) to optimize
            lr (float, optional): learning rate; defaults to 1e-3
            beta1 (optional):
                coefficient used for computing the running average; defaults to 0.9
            beta2 (optional):
                coefficient to compute the squared running average; defaults to 0.999
            epsilon (optional):
                term added to the denominator to avoid division bt 0; defaults to 1e-8
        """
        super(Adam, self).__init__(params, lr)
        self.__betas = (beta1, beta2)
        self.__epsilon = epsilon
        self.__exp_avgs = [torch.zeros_like(parameter) for parameter in params]
        self.__squared_exp_avgs = [torch.zeros_like(parameter) for parameter in params]

    @torch.no_grad()
    def step(self):
        i = 0
        for parameter, exp_avg, squared_exp_avg in zip(self._params, self.__exp_avgs,
                                                       self.__squared_exp_avgs):
            if parameter.grad is not None:
                grad = parameter.grad

                beta_1, beta_2 = self.__betas

                self._step[i] += 1

                torch.add(exp_avg * beta_1, grad, alpha=1 - beta_1, out=exp_avg)
                torch.addcmul(squared_exp_avg * beta_2, grad, grad, value=1 - beta_2,
                              out=squared_exp_avg)

                bias_correction1 = 1 - beta_1 ** self._step[i]
                bias_correction2 = 1 - beta_2 ** self._step[i]

                denominator = (torch.sqrt(squared_exp_avg) / math.sqrt(
                    bias_correction2)) + self.__epsilon
                step_size = self._learning_rate / bias_correction1

                torch.addcdiv(parameter, exp_avg, denominator, value=-step_size,
                              out=parameter)
                i += 1
