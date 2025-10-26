from types import NoneType

import torch
from torch import Tensor
from torch import nn

from . import NCSN


class ScoreNet(nn.Module):
    def __init__(self, input_dim: torch.Size | tuple[int, int, int], activation_function: nn.Module | NoneType = None, filters: int = 128):
        super().__init__()

        self._ncsn = NCSN(input_dim, 1, activation_function, filters)

    def forward(self, x: Tensor):
        return self._ncsn(x, 0)
