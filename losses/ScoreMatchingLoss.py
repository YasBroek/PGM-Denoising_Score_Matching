from torch import Tensor
from torch import nn

from . import Perturbation


class ScoreMatchingLoss(nn.Module):
    def __init__(self, perturbation: Perturbation, *params):
        super().__init__()

        self.perturbation = perturbation
        self.params = params

    def forward(self, x: Tensor, score: nn.Module):
        B = x.shape[0]
        x_noisy = self.perturbation.noise(x, *self.params)

        pred = score(x_noisy).view(B, -1)
        target = self.perturbation.score(x, x_noisy, *self.params).view(B, -1)

        loss = 0.5 * ((pred - target) ** 2).sum(dim=-1)
        return loss
