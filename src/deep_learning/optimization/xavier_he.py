import math
from typing import Optional

import torch
from torch import Tensor

from autocorrect import corrector, token


def xavier_init(first_dim: int, second_dim: int, r: Optional[Tensor] = None,
                factor: float = 1) -> Tensor:
    """
    Implementation of the Xavier weights initialization as proposed by Xavier Glorot and Yoshua
    Bengio on 'Understanding the difficulty of training deep feedforward neural networks'

    Args:
        first_dim:
            the first dimension of the weights tensor
        second_dim:
            the second dimension of the weights tensor
        r (optional):
            a tensor of random numbers of the same dimensions as the weights tensor
        factor (optional):
            a scalar to be used in variations of the Xavier initialization; see ``he_init``
    Returns:
        the initialized weights tensor
    """
    factor *= 1 / first_dim
    return (r if r is not None else torch.randn((first_dim, second_dim))) * math.sqrt(factor)


def he_init(first_dim: int, second_dim: int, r: Optional[Tensor] = None) -> Tensor:
    """
    Implementation of the He weights initialization as proposed by Kaiming He et.al. on 'Delving
    Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification'
    (arXiv:1502.01852 [cs.CV])

    Args:
        first_dim:
            the first dimension of the weights tensor
        second_dim:
            the second dimension of the weights tensor
        r (optional):
            a tensor of random numbers of the same dimensions as the weights tensor
    Returns:
        the initialized weights tensor
    """
    return xavier_init(first_dim, second_dim, r, 2)


if __name__ == '__main__':
    # Tests del API del curso
    r_xavier = corrector.get_test_data(homework=3, question="2a", test=1, token=token)
    r_he = corrector.get_test_data(homework=3, question="2a", test=2, token=token)

    w_xavier = xavier_init(50, 50, torch.tensor(r_xavier))
    w_he = he_init(50, 50, torch.tensor(r_he))

    corrector.submit(homework=3, question="2a", test=1, token=token, answer=w_xavier)
    corrector.submit(homework=3, question="2a", test=2, token=token, answer=w_he)
