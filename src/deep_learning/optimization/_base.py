"""
"optimally-regular" (c) by Ignacio Slater M.
"optimally-regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""

from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class AbstractOptimizer(ABC):
    """Base class for all optimizers.
    """
    _learning_rate: float
    _params: List[Tensor]

    def __init__(self, params: List[Tensor], lr: float):
        """Initializes the common parameters of all optimizers."""
        self._params = params
        self._learning_rate = lr
        self._step = [0] * len(params)

    @abstractmethod
    def step(self):
        """"Performs a single optimization step (parameter update)."""
        raise NotImplementedError
