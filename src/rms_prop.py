"""
"Back to training" (c) by Ignacio Slater M.
"Back to training" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""


from typing import Iterable


class RMSProp():
    """Root mean square propagation implementation.
    """

    __lr: float
    __beta: float
    __epsilon: float
    __parameters: Iterable

    def __init__(self, parameters, lr=0.001, beta=0.9, epsilon=1e-8):
        """
        Implementation of the RMSprop algorithm as proposed by G. Hinton in his
        course
        <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>.

        Args:
            parameters (iterable):
                iterable of parameters to optimize
            lr (float, optional):
                the learning rate; defaults to 0.001
            beta (float, optional):
                the smoothing constant; defaults to 0.9
            epsilon ([type], optional):
                term added to the denominator to improve numerical stability; 
                defaults to 1e-8
        """
        self.__lr = lr
        self.__beta = beta
        self.__epsilon = epsilon
        self.__parameters = parameters

    def step():
        # actualiza acá los parámetros a partir de los gradientes
        # y la corrección según S
        pass
