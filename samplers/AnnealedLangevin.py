from types import NoneType

import torch
from torch import Tensor
from torch import nn

from . import LangevinDynamics


class AnnealedLangevinDynamics:
    def __init__(self, score: nn.Module, sigmas: Tensor, device: str = "cpu"):
        self.device = device
        self.score = score.to(device)
        self.sigmas = sigmas.to(device)

        self.L = sigmas.size(dim=0)

    def sample(self, shape, x0: Tensor | NoneType = None, T: int = 10, epsilon: float = 1e-6, return_all_samples: bool = False):
        if x0 is None:
            x0 = torch.rand(shape).to(self.device)

        x = x0
        all_samples = [x0]

        for i in range(self.L):
            score_sigma = nn.Module(lambda x_in: self.score(x_in, self.sigmas[i]))
            sampler = LangevinDynamics(score_sigma, self.device)

            step_size = epsilon * (self.sigmas[i] / self.sigmas[-1]) ** 2
            x = sampler.sample(shape, x, T, step_size, False)

        if return_all_samples:
            return all_samples

        return x
