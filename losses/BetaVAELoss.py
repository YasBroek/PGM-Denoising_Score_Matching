import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class BetaVAELoss(nn.Module):
    def __init__(self, beta):
        super().__init__()

        self.beta = beta

    def _reconstruction_loss(self, x_hat: Tensor, x: Tensor):
        loss = F.mse_loss(x_hat, x, reduction="sum")
        batch_size = x.shape[0]

        return loss / batch_size

    def _kl_normal_loss(self, mean: Tensor, logvar: Tensor):
        var = torch.exp(logvar)
        kl_per_sample = 0.5 * (mean**2 + var - 1 - logvar).sum(dim=1)

        return kl_per_sample.mean()

    def forward(self, x_hat: Tensor, x: Tensor, params_z: list[Tensor]):
        rec_loss = self._reconstruction_loss(x_hat, x)
        kl_loss = self._kl_normal_loss(*params_z)

        loss = rec_loss + self.beta * kl_loss
        return loss
