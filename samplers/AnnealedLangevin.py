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

    def sample(
        self,
        arg: Tensor | Size | tuple | list | NoneType = None,
        T: int = 100,
        epsilon: float = 2e-5,
        return_all_samples: bool = False,
    ):
        if arg is None:
            raise ValueError("Specify either shape or x.")

        if isinstance(arg, (Size, tuple, list)):
            shape = arg
            x = torch.rand(shape).to(self.device)
        elif isinstance(arg, Tensor):
            x = arg.to(self.device)
            shape = x.shape

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
