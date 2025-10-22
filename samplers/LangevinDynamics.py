import torch
from torch import Tensor
from torch import nn


class LangevinDynamics:
    def __init__(self, score: nn.Module, device: str = "cpu"):
        self.device = device
        self.score = score.to(self.device)

    def sample(self, shape, x0: Tensor, T: int = 10, step_size: float | Tensor = 1e-6, return_all_samples: bool = False):
        self.score.eval()
        step_size = Tensor(step_size, device=self.device)

        if not x0:
            x0 = torch.rand(shape).to(self.device)

        x = torch.zeros((T, *shape), device=self.device)
        x[0] = x0

        for t in range(T):
            z_t = torch.randn_like(x0)
            x_t = x[t - 1] + 0.5 * step_size * self.score(x[t - 1]) + torch.sqrt(step_size) * z_t

            x[t] = x_t

        if return_all_samples:
            return x

        return x[-1]
