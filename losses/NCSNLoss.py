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

        labels = torch.randint(0, L, (B,), device=x.device)
        used_sigmas = self.sigmas[labels].view(-1, *([1] * (x.dim() - 1)))

        sm_loss = ScoreMatchingLoss(self.perturbation, used_sigmas)
        score_sigma = LambdaModule(lambda x_in: score(x_in, labels))

        loss = self.coeff_func(used_sigmas.view(B)) * sm_loss(x, score_sigma)
        return loss.mean()
