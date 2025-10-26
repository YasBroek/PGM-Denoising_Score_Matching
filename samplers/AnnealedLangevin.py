from types import NoneType

import torch
from torch import Tensor
from torch import nn

from . import LangevinDynamics
from utils import LambdaModule


class AnnealedLangevinDynamics:
    def __init__(self, score: nn.Module, sigmas: Tensor, device: torch.device | None = None):
        self.device = device

        if device is not None:
            self.score = score.to(device)
            self.sigmas = sigmas.to(device)

        self.L = sigmas.size(dim=0)

    def sample(self, shape, x0: Tensor | NoneType = None, T: int = 100, epsilon: float = 2e-5, return_all_samples: bool = False):
        if x0 is None:
            if self.device is None:
                self.device = torch.device("cpu")

            x0 = torch.rand(shape).to(self.device)

        if self.device is None:
            self.device = x0.device

        self.score = self.score.to(self.device)
        self.sigmas = self.sigmas.to(self.device)

        x = x0
        all_samples = [x]

        for i in range(self.L):
            score_sigma = LambdaModule(lambda x_in: self.score(x_in, torch.tensor(i, device=self.device)))
            sampler = LangevinDynamics(score_sigma, self.device)

            step_size = epsilon * (self.sigmas[i] / self.sigmas[-1]) ** 2

            x = sampler.sample(shape, x, T, step_size)
            all_samples.append(x)

        if return_all_samples:
            return all_samples

        return x
