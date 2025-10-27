import torch
from torch import nn


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.version.hip and torch.version.hip != "":  # ROCm (AMD)
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ConditionalSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        self.outputs = []
        self.shapes = []

        for module in self:
            try:
                x = module(x, *args, **kwargs)
            except TypeError:
                x = module(x)

            self.outputs.append(x)
            self.shapes.append(x.shape)

        return x
