import math
from typing import List, Tuple

import torch
from torch import Tensor

from deep_learning.optimization._base import AbstractOptimizer


class Adam(AbstractOptimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        epsilon (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    __betas: Tuple[float, float]
    __epsilon: float
    __exp_avgs: List[Tensor]
    __squared_exp_avgs: List[Tensor]

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0, amsgrad=False):
        self.__betas = (beta1, beta2)
        self.__epsilon = epsilon
        defaults = dict(lr=lr, betas=self.__betas, eps=epsilon,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, lr, defaults)
        self.__exp_avgs = [torch.zeros_like(parameter) for parameter in params]
        self.__squared_exp_avgs = [torch.zeros_like(parameter) for parameter in params]

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        for group in self.param_groups:
            i = 0
            for parameter, exp_avg, squared_exp_avg in zip(self._params, self.__exp_avgs,
                                                           self.__squared_exp_avgs):
                if parameter.grad is None:
                    continue
                grad = parameter.grad

                beta1, beta2 = group['betas']

                self._step[i] += 1
                bias_correction1 = 1 - beta1 ** self._step[i]
                bias_correction2 = 1 - beta2 ** self._step[i]

                if group['weight_decay'] != 0:
                    grad = grad.add(parameter, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                squared_exp_avg.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (squared_exp_avg.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                parameter.addcdiv_(exp_avg, denom, value=-step_size)
                i += 1

        return loss
