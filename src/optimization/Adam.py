"""
"Optimally regular" (c) by Ignacio Slater M.
"Optimally regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""


class Adam():
    """Implementation of the Adam algorithm.
    """
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Implements the Adam algorithm as proposed by Kingma et.al. 
        (arXiv:1412.6980)

        Args:
            parameters (Tensor): 
                the parameters to optimize
            lr (float, optional): 
                the learning rate; defaults to 0.001
            beta1 (float, optional): 
                coefficient used for computing running averages; defaults to 
                0.9
            beta2 (float, optional): 
                coefficient for running gradient squares; defaults to 0.999
            epsilon (float, optional): 
                term added to the denominator to improve numerical stability; 
                defaults to 1e-8
        """
        pass

    def step(self):
        # actualiza acá los parámetros a partir de los gradientes
        # y la corrección según S
        pass
