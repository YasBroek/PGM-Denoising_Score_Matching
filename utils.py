import copy

import torch
from torch import nn
from torch.nn import functional as F


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.version.hip and torch.version.hip != "":  # ROCm (AMD)
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def compute_mean_std_channels(train_loader, encoder, device: torch.device = get_torch_device()):
    channel_sum = None
    channel_sq_sum = None
    num_elements = 0

    with torch.no_grad():
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)
            z = encoder(x)

            B, C, H, W = z.shape
            z_flat = z.view(B, C, -1)

            if channel_sum is None:
                channel_sum = torch.zeros(C, device=device)
                channel_sq_sum = torch.zeros(C, device=device)

            channel_sum += z_flat.sum(dim=(0, 2))
            channel_sq_sum += (z_flat**2).sum(dim=(0, 2))
            num_elements += B * H * W

    if channel_sum is None or channel_sq_sum is None:
        raise RuntimeError("Channel sum or channel squared sum is None.")

    mean = channel_sum / num_elements
    var = channel_sq_sum / num_elements - mean**2
    std = torch.sqrt(var + 1e-8)

    return mean[:, None, None], std[:, None, None]


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
            activation = nn.LeakyReLU()

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, activation=None, use_bn=False):
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = F.silu(out)
        return out
