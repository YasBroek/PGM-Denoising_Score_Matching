from typing import Callable

import torch
from torch import Tensor
from torch import nn

from utils import LambdaModule

from . import ScoreMatchingLoss


class NCSNLoss(nn.Module):
    def __init__(self, coeff_func: Callable, perturbation="gaussian", K: int = 6, **kwargs):
        super().__init__()

        self.coeff_func = coeff_func
        self.perturbation = perturbation
        self.K = K

        self.kwargs = kwargs

    def _gaussian_noise(self, x: Tensor, score: nn.Module):
        sigmas = self.kwargs.get("sigmas", [])
        L = len(sigmas)
        B = x.size(0)

        loss = 0

        indices = torch.randperm(L, device=x.device)[:self.K]
        used_sigmas = sigmas[indices]

        for i, sigma in zip(indices, used_sigmas):
            labels = torch.full((B,), i.item(), dtype=torch.long, device=x.device)

            sm_loss = ScoreMatchingLoss("gaussian", sigma=sigma)
            score_sigma = LambdaModule(lambda x_in: score(x_in, labels))

            loss += self.coeff_func(sigma) * sm_loss(x, score_sigma)

        return loss / self.K

    def forward(self, x: Tensor, score: nn.Module):
        if self.perturbation == "gaussian":
            return self._gaussian_noise(x, score)

        return 0
