"""
"Optimally Regular" (c) by Ignacio Slater M.
"Optimally Regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from torch import Tensor


def ce_loss(q: Tensor, p: Tensor, stable=True, epsilon=1e-7):
    """Cross entropy loss function."""
    if stable:
        q = q.clamp(epsilon, 1 - epsilon)
    return -(p * q.log()).sum() / q.size()[0]
