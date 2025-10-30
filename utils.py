import copy

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


class MLPBlock(nn.Module):
    def __init__(self, neurons, activation: nn.Module | None = None, use_bn: bool = False, dropout: float = 0.0, last_activation: bool = False):
        super().__init__()

        if len(neurons) < 2:
            raise ValueError("`neurons` must be a list/tuple like [in_features, ..., out_features] with length >= 2.")

        if activation is None:
            activation = nn.ReLU()

        layers: list[nn.Module] = []
        num_layers = len(neurons) - 1

        for i in range(num_layers):
            in_f, out_f = neurons[i], neurons[i + 1]
            layers.append(nn.Linear(in_f, out_f))

            if last_activation or i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_f))

                layers.append(copy.deepcopy(activation))

                if dropout and dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
