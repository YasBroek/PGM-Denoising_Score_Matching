from types import NoneType
import torch
from torch import Tensor
from torch import nn


class LangevinDynamics:
    def __init__(self, score: nn.Module, device: torch.device = torch.device("cpu")):
        self.device = device
        self.score = score.to(device)

    def sample(self, shape, x0: Tensor | NoneType = None, T: int = 10, step_size: float | Tensor = 1e-6) -> Tensor:
        self.score.eval()

        step_size = torch.as_tensor(step_size, device=self.device)

        if x0 is None:
            x = torch.rand(shape).to(self.device)
        else:
            x = x0

        with torch.no_grad():
            for _ in range(T):
                z_t = torch.randn_like(x).to(self.device)
                x = x + 0.5 * step_size * self.score(x) + torch.sqrt(step_size) * z_t

        return x
