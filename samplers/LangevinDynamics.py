from types import NoneType
import torch
from torch import Tensor, Size
from torch import nn

from utils import get_torch_device


class LangevinDynamics:
    def __init__(self, score: nn.Module, device: torch.device = get_torch_device()):
        self.score = score.to(device)
        self.device = device

    def sample(self, arg: Tensor | Size | tuple | list | NoneType = None, T: int = 10, step_size: float | Tensor = 1e-6) -> Tensor:
        self.score.eval()

        if arg is None:
            raise ValueError("Specify either shape or x.")

        if isinstance(arg, (Size, tuple, list)):
            shape = arg
            x = torch.rand(shape).to(self.device)
        elif isinstance(arg, Tensor):
            x = arg.to(self.device)
            shape = x.shape

        step_size = torch.as_tensor(step_size, device=self.device)

        with torch.no_grad():
            for _ in range(T):
                z_t = torch.randn_like(x).to(self.device)
                x = x + 0.5 * step_size * self.score(x) + torch.sqrt(step_size) * z_t

        return x
