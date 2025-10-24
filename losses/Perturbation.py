import torch
from torch import Tensor


class Perturbation:
    def noise(self, x: Tensor, *params):
        raise NotImplementedError

    def score(self, x: Tensor, x_noisy: Tensor, *params):
        raise NotImplementedError


class GaussianPerturbation(Perturbation):
    def noise(self, x: Tensor, *params):
        sigma = params[0]

        noise = torch.randn_like(x)
        x_noisy = sigma * noise + x

        return x_noisy

    def score(self, x: Tensor, x_noisy: Tensor, *params):
        sigma = params[0]
        return - (x_noisy - x) / (sigma ** 2)
