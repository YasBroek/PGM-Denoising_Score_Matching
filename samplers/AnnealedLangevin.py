from types import NoneType

import torch
from torch import Tensor, Size
from torch import nn

from . import LangevinDynamics
from utils import get_torch_device, LambdaModule


class AnnealedLangevinDynamics:
    def __init__(self, score: nn.Module, sigmas: Tensor, device: torch.device = get_torch_device()):
        self.score = score.to(device)
        self.sigmas = sigmas.to(device)
        self.device = device

        self.L = sigmas.size(dim=0)

    def _sample(
        self,
        x: Tensor,
        T: int = 100,
        epsilon: float = 2e-5,
        return_all_samples: bool = False,
    ):
        self.score = self.score.to(self.device)
        self.sigmas = self.sigmas.to(self.device)

        all_samples = [x]

        for i in range(self.L):
            score_sigma = LambdaModule(lambda x_in: self.score(x_in, torch.tensor(i, device=self.device)))
            sampler = LangevinDynamics(score_sigma, self.device)

            step_size = epsilon * (self.sigmas[i] / self.sigmas[-1]) ** 2

            x = sampler.sample(x, T, step_size)
            all_samples.append(x)

        if return_all_samples:
            return all_samples

        return x

    def sample_from_tensor(self, x: Tensor, N: int | NoneType = None, mean: Tensor | NoneType = None, std: Tensor | NoneType = None, T: int = 100, epsilon: float = 2e-5, return_all_samples: bool = False):
        if N is not None and N <= 0:
            raise ValueError("Number of samples cannot be negative.")

        if N is not None:
            x = x.unsqueeze(0).repeat(N, *([1] * x.dim()))

        if mean is None:
            mean = torch.zeros_like(x)
        elif mean.dim() == x.dim() - 1:
            mean = mean[None, ...]
        else:
            raise ValueError("Mean must have the same dimension as the samples.")

        if std is None:
            std = torch.ones_like(x)
        elif std.dim() == x.dim() - 1:
            std = std[None, ...]
        else:
            raise ValueError("Std must have the same dimension as the samples.")

        y = self._sample(x, T, epsilon, return_all_samples)

        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = std * y[i] + mean
        else:
            y = std * y + mean

        return y

    def sample(self, shape: Size | tuple | list, N: int | NoneType = None, mean: Tensor | NoneType = None, std: Tensor | NoneType = None, T: int = 100, epsilon: float = 2e-5, return_all_samples: bool = False):
        if N is not None and N <= 0:
            raise ValueError("Number of samples cannot be negative.")

        if N is not None:
            shape = (N, *shape)

        x = torch.rand(shape).to(self.device)

        return self.sample_from_tensor(x, None, mean, std, T, epsilon, return_all_samples)
