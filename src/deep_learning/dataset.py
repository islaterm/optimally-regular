"""
"optimally-regular" (c) by Ignacio Slater M.
"optimally-regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, N, F, C):
        self.N = N
        self.data = torch.rand(N, F)
        self.targets = torch.randint(0, C, (N,))

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


class BernuliDataset(Dataset):
    def __init__(self, N, F, C):
        self.N = N
        self.data = torch.bernoulli(torch.rand(N, F))
        self.targets = torch.randint(0, C, (N,))

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


class ToyDataset(Dataset):
    def __init__(self, logic_gate='xor'):
        self.N = 4
        self.data = torch.tensor([[1., 1.], [1., 0.], [0., 1.], [0., 0.]])
        if logic_gate == 'XOR':
            self.targets = torch.tensor([0, 1, 1, 0])
        elif logic_gate == 'AND':
            self.targets = torch.tensor([1, 0, 0, 0])
        elif logic_gate == 'OR':
            self.targets = torch.tensor([1, 1, 1, 0])

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.data[i], self.targets[i]
