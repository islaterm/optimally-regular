"""
"Optimally regular" (c) by Ignacio Slater M.
"Optimally regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
import torch


class SGD:
    """Implementation of a Stochastic Gradient Descent algorithm."""
    __learning_rate: float
    __parameters: torch.nn.ParameterList
    __weight_decay: float

    def __init__(self, parameters: torch.nn.ParameterList, lr: float, weight_decay=0):
        """
        Initializes a new SGD object.

        Args:
            parameters:
                the parameters of the neural network to be optimized
            lr:
                the learning rate of the algorithm
            weight_decay:
                The value used for a weight decay regularization of the SGD.
        """
        self.__parameters = parameters
        self.__learning_rate = lr
        self.__weight_decay = weight_decay

    def step(self) -> None:
        """Updates the parameters according to their respective gradients."""
        param: torch.nn.Parameter
        for param in self.__parameters:
            param.data -= self.__learning_rate * (param.grad + self.__weight_decay * param.data)
