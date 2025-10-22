import torch
from torch import Tensor
from torch import nn

from . import ScoreMatchingLoss


class NCSNLoss(nn.Module):
    def __init__(self, coeff_func: function, perturbation="gaussian", **kwargs):
        sm_loss = ScoreMatchingLoss(perturbation, **kwargs)
        self.coeff_func = coeff_func

    def forward(self, x: Tensor, score: nn.Module):
        pass
