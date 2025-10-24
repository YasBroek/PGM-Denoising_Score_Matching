from typing import Callable

import torch
from torch import Tensor
from torch import nn

from utils import LambdaModule

from . import ScoreMatchingLoss, Perturbation


class NCSNLoss(nn.Module):
    def __init__(self, perturbation: Perturbation, sigmas: Tensor, coeff_func: Callable, K: int = 6):
        super().__init__()

        self.perturbation = perturbation
        self.coeff_func = coeff_func
        self.sigmas = sigmas

        self.K = K

    def forward(self, x: Tensor, score: nn.Module):
        L = len(self.sigmas)
        B = x.size(0)

        loss = 0

        indices = torch.randperm(L, device=x.device)[: self.K]
        used_sigmas = self.sigmas[indices]

        for i, sigma in zip(indices, used_sigmas):
            labels = torch.full((B,), i.item(), dtype=torch.long, device=x.device)

            sm_loss = ScoreMatchingLoss(self.perturbation, sigma)
            score_sigma = LambdaModule(lambda x_in: score(x_in, labels))

            loss += self.coeff_func(sigma) * sm_loss(x, score_sigma)

        return loss / self.K
