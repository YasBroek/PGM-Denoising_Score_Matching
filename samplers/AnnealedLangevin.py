import torch
from torch import Tensor
from torch import nn

from . import LangevinSampler, LangevinDynamics
from utils import get_torch_device, LambdaModule


class AnnealedLangevinDynamics(LangevinSampler):
    def __init__(self, score: nn.Module, sigmas: Tensor, device: torch.device = get_torch_device()):
        super().__init__(score, device)

        self.sigmas = sigmas.to(device)
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
            step_size = epsilon * (self.sigmas[i] / self.sigmas[-1]) ** 2

            score_sigma = LambdaModule(lambda x_in: self.score(x_in, torch.tensor(i, device=self.device)))
            sampler = LangevinDynamics(score_sigma, step_size, self.device)

            x = sampler.sample_from_tensor(x, T, epsilon)  # pyright: ignore[reportAssignmentType]
            all_samples.append(x)

        if return_all_samples:
            return all_samples

        return x
