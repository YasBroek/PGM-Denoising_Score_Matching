import torch
from torch import nn
import torch.nn.functional as F

from utils import MLPBlock, ConvBlock


class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, activation=None, use_bn=False):
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=16, conv_layers=2, base_channels=32, mlp_layers=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.mlp = mlp_layers is not None and len(mlp_layers) > 0
        in_channels = input_shape[0]

        self.conv_block = nn.Sequential(
            ConvBlock(in_channels, base_channels, 4, stride=2, padding=1),
            *[ConvBlock(base_channels, base_channels, 4, stride=2, padding=1) for _ in range(conv_layers)],
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv_block(dummy)

        C_out, Hc, Wc = conv_out.shape[1:]

        if self.mlp:
            flattened_dim = C_out * Hc * Wc

            self.end_block = MLPBlock([flattened_dim, *mlp_layers, latent_dim])
            self.conv_shape = (C_out, Hc, Wc)
            self.z_shape = (latent_dim,)
        else:
            self.end_block = nn.Conv2d(base_channels, latent_dim, kernel_size=1)
            self.conv_shape = (latent_dim, Hc, Wc)
            self.z_shape = self.conv_shape

    def forward(self, x):
        squeeze = False

        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True

        x = self.conv_block(x)

        if self.mlp:
            x = x.flatten(start_dim=1)

        x = self.end_block(x)

        if squeeze:
            x = x.squeeze(0)

        return x


class Decoder(nn.Module):
    def __init__(self, conv_shape, out_shape, latent_dim=16, convT_layers=2, base_channels=32, mlp_layers=None):
        super().__init__()

        self.conv_shape = conv_shape
        self.mlp = mlp_layers is not None and len(mlp_layers) > 0
        self.output_shape = out_shape

        out_channels = out_shape[0]

        if self.mlp:
            flattened_dim = int(torch.prod(torch.tensor(conv_shape)))
            self.start_block = MLPBlock([latent_dim, *mlp_layers, flattened_dim])
            first_in_channels = conv_shape[0]
        else:
            self.start_block = nn.Conv2d(latent_dim, base_channels, kernel_size=1)
            first_in_channels = base_channels

        convT_blocks = [ConvTBlock(first_in_channels, base_channels, 4, stride=2, padding=1)]
        for _ in range(convT_layers - 1):
            convT_blocks.append(ConvTBlock(base_channels, base_channels, 4, stride=2, padding=1))
        convT_blocks.append(ConvTBlock(base_channels, out_channels, 4, stride=2, padding=1, activation=nn.Sigmoid(), use_bn=False))

        self.convT_block = nn.Sequential(*convT_blocks)

    def forward(self, z):
        squeeze = False

        if (self.mlp and z.dim() == 1) or (not self.mlp and z.dim() == 3):
            z = z.unsqueeze(0)
            squeeze = True

        batch_size = z.size(0)

        z = self.start_block(z)
        if self.mlp:
            z = z.view(batch_size, *self.conv_shape)

        z = self.convT_block(z)

        H_t, W_t = self.output_shape[1], self.output_shape[2]
        if z.shape[-2] != H_t or z.shape[-1] != W_t:
            z = F.interpolate(z, size=(H_t, W_t), mode="bilinear", align_corners=False)

        if squeeze:
            z = z.squeeze(0)

        return z
