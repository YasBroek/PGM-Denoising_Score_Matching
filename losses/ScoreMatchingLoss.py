import torch
from torch import Tensor
from torch import nn

from . import Perturbation


class ScoreMatchingLoss(nn.Module):
    def __init__(self, perturbation: Perturbation, *params):
        super().__init__()

        self.perturbation = perturbation
        self.params = params

    def forward(self, x: Tensor, score: nn.Module):
        x_noisy = self.perturbation.noise(x, *self.params)

        out = score(x_noisy)
        target = self.perturbation.score(x, x_noisy, *self.params)

        loss = 0.5 * torch.square(out - target).mean()

        return loss
