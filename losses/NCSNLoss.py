from typing import Callable

import torch
from torch import Tensor
from torch import nn

from . import ScoreMatchingLoss


class NCSNLoss(nn.Module):
    def __init__(self, coeff_func: Callable, perturbation="gaussian", **kwargs):
        self.coeff_func = coeff_func
        self.perturbation = perturbation
        self.kwargs = kwargs

    def _gaussian_noise(self, x: Tensor, score: nn.Module):
        sigmas = self.kwargs.get("sigmas", [])
        L = len(sigmas)

        loss = 0

        for sigma in sigmas:
            sm_loss = ScoreMatchingLoss("gaussian", sigma=sigma)
            score_sigma = nn.Module(lambda x_in: score(x_in, sigma))

            loss += self.coeff_func(sigma) * sm_loss(x, score_sigma)

        return loss / L

    def forward(self, x: Tensor, score: nn.Module):
        if self.pertubation == "gaussian":
            return self._gaussian_noise(x, score)

        return 0
