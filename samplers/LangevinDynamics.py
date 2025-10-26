from types import NoneType
import torch
from torch import Tensor
from torch import nn


class LangevinDynamics:
    def __init__(self, score: nn.Module, device: torch.device = torch.device("cpu")):
        self.device = device
        self.score = score.to(self.device)

    def sample(self, shape, x0: Tensor | NoneType = None, T: int = 10, step_size: float | Tensor = 1e-6, return_all_samples: bool = False):
        self.score.eval()
        step_size = torch.as_tensor(step_size, device=self.device)

        with torch.no_grad():
            if x0 is None:
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
