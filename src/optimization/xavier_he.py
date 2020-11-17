import torch
from torch import Tensor
from torch import Tensor
from torch import nn

nn.init.xavier_normal
def xavier_he_init(weights_tensor: Tensor):
  prev_d = weights_tensor.size()[1]
