import torch
from torch import nn

from utils import MLPBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, activation=None, use_bn=False):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, activation=None, use_bn=False):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=12, multiplier=1, conv_layers=2, mlp_layers=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.multiplier = multiplier

        in_channels = input_shape[0]

        self.conv_block = nn.Sequential(
            ConvBlock(in_channels, 32, 4, stride=1, padding=1),
            *[ConvBlock(32, 32, 4, stride=1) for _ in range(conv_layers)],
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv_block(dummy)

        self.conv_shape = conv_out.shape[1:]
        self.flattened_dim = int(torch.prod(torch.tensor(self.conv_shape)))

        if mlp_layers is None:
            mlp_layers = [64]

        self.mlp_block = MLPBlock([self.flattened_dim, *mlp_layers, latent_dim * multiplier])

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv_block(x)
        x = x.flatten(start_dim=1)

        x = self.mlp_block(x)

        if self.multiplier > 1:
            x = x.view(batch_size, self.latent_dim, self.multiplier)

        return x


class Decoder(nn.Module):
    def __init__(self, conv_shape, out_channels, latent_dim=12, convT_layers=2, mlp_layers=None):
        super().__init__()

        self.conv_shape = conv_shape
        flattened_dim = int(torch.prod(torch.tensor(conv_shape)))

        if mlp_layers is None:
            mlp_layers = [64]

        self.mlp_block = MLPBlock([latent_dim, *mlp_layers, flattened_dim])

        self.convT_block = nn.Sequential(
            *[ConvTBlock(32, 32, 4, stride=1) for _ in range(convT_layers)],
            ConvTBlock(32, out_channels, 4, stride=1, padding=1, activation=nn.Sigmoid(), use_bn=False),
        )

    def forward(self, z):
        batch_size = z.size(0)

        z = self.mlp_block(z)

        z = z.view(batch_size, *self.conv_shape)
        z = self.convT_block(z)

        return z
