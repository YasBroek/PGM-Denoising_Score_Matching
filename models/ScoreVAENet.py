import copy
from types import NoneType

import torch
from torch import Tensor
from torch import nn

from utils import ConditionalSequential


class ConditionalLayerNorm1d(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.embed = nn.Embedding(num_classes, hidden_dim * 2)
        self.embed.weight.data[:, :hidden_dim].fill_(1.0)
        self.embed.weight.data[:, hidden_dim:].zero_()

    def forward(self, x, labels):
        h = self.norm(x)
        gamma, beta = self.embed(labels).chunk(2, dim=-1)
        return gamma * h + beta


class ConditionalMLPBlock(nn.Module):
    def __init__(self, neurons, num_cls: int, activation: nn.Module | None = None, dropout: float = 0.0, last_activation: bool = False):
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
                layers.append(ConditionalLayerNorm1d(out_f, num_cls))

                layers.append(copy.deepcopy(activation))

                if dropout and dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.block = ConditionalSequential(*layers)

    def forward(self, x, y):
        return self.block(x, y)


class ScoreVAENet(nn.Module):
    def __init__(
        self,
        input_size: torch.Size | int,
        num_cls: int,
        activation_function: nn.Module | NoneType = None,
        filters: int = 128,
        size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        neurons = [input_size]

        neurons.extend([filters * (2**i) for i in range(size)])
        neurons.extend([filters * (2**i) for i in range(size - 1, -1, -1)])

        neurons.append(input_size)

        self.mlp = ConditionalMLPBlock(neurons, num_cls, activation_function, dropout, False)

    def forward(self, x: Tensor, y: Tensor):
        x = x.view(x.size(0), -1)
        return self.mlp(x, y)
