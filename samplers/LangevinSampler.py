from types import NoneType

import torch
from torch import Tensor, Size
from torch import nn

from utils import get_torch_device


class LangevinSampler:
    def __init__(self, score: nn.Module, device: torch.device = get_torch_device()):
        self.score = score.to(device)
        self.device = device

    def _sample(self, x: Tensor, T: int = 100, epsilon: float = 2e-5, return_all_samples: bool = False):
        raise NotImplementedError()

    def sample_from_tensor(
        self,
        x: Tensor,
        T: int = 100,
        epsilon: float = 2e-5,
        mean: Tensor | NoneType = None,
        std: Tensor | NoneType = None,
        N: int | NoneType = None,
        return_all_samples: bool = False,
    ) -> Tensor | list[Tensor]:
        if N is not None and N <= 0:
            raise ValueError("Number of samples cannot be negative.")

        if N is not None:
            x = x.unsqueeze(0).repeat(N, *([1] * x.dim()))

        if mean is None:
            mean = torch.zeros_like(x)
        elif mean.dim() == x.dim() - 1:
            mean = mean[None, ...]
        elif mean.dim() != x.dim():
            raise ValueError("Mean must have the same dimension as the samples.")

        if std is None:
            std = torch.ones_like(x)
        elif std.dim() == x.dim() - 1:
            std = std[None, ...]
        elif std.dim() != x.dim():
            raise ValueError("Std must have the same dimension as the samples.")

        y = self._sample(x, T, epsilon, return_all_samples)

        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = std * y[i] + mean
        else:
            y = std * y + mean

        return y

    def sample(
        self,
        shape: Size | tuple | list,
        T: int = 100,
        epsilon: float = 2e-5,
        mean: Tensor | NoneType = None,
        std: Tensor | NoneType = None,
        N: int | NoneType = None,
        return_all_samples: bool = False,
    ) -> Tensor | list[Tensor]:
        if N is not None and N <= 0:
            raise ValueError("Number of samples cannot be negative.")

        if N is not None:
            shape = (N, *shape)

        x = torch.rand(shape).to(self.device)

        return self.sample_from_tensor(x, T, epsilon, mean, std, None, return_all_samples)
