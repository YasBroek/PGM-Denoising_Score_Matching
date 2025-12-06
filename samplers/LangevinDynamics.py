import torch
from torch import Tensor
from torch import nn

from . import LangevinSampler
from utils import get_torch_device


class LangevinDynamics(LangevinSampler):
    def __init__(self, score: nn.Module, step_size: float | Tensor = 1e-6, device: torch.device = get_torch_device()):
        super().__init__(score, device)

        self.step_size = torch.as_tensor(step_size, device=self.device)

    def _sample(self, x: Tensor, T: int = 100, epsilon: float = 2e-5, return_all_samples: bool = False) -> Tensor:
        self.score.eval()

        with torch.no_grad():
            for _ in range(T):
                z_t = torch.randn_like(x).to(self.device)
                x = x + 0.5 * self.step_size * self.score(x) + torch.sqrt(self.step_size) * z_t

        return x
