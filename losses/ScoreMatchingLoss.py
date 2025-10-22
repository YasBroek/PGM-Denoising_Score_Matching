import torch
from torch import Tensor
from torch import nn


class ScoreMatchingLoss(nn.Module):
    def __init__(self, perturbation="gaussian", **kwargs):
        super().__init__()

        if perturbation == "gaussian":
            self.q = self._gaussian_noise
            self.score_q = self._gaussian_score_q

        self.kwargs = kwargs

    def _gaussian_noise(self, x: Tensor):
        noise = torch.randn_like(x)
        sigma = self.kwargs.get("sigma", 1)

        x_noisy = sigma * noise + x
        return x_noisy

    def _gaussian_score_q(self, x, x_noisy):
        sigma = Tensor(self.kwargs.get("sigma", 1), device=x.device)

        return -(x_noisy - x) / (sigma**2)

    def forward(self, x: Tensor, score: nn.Module):
        x_noisy = self.q(x)

        out = score(x_noisy)
        target = self.score_q(x, x_noisy)

        loss = 0.5 * torch.square(out - target).mean()

        return loss
