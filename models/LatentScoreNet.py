import torch
from torch import nn
from torch.nn import functional as F

from utils import ResidualBlock


class LatentScoreNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        L: int,
        base_channels: int = 64,
        res_blocks: int = 4,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.L = L

        self.label_embed = nn.Embedding(L, 1)

        self.in_conv = nn.Conv2d(latent_dim + 1, base_channels, kernel_size=3, padding=1)

        blocks = []
        for _ in range(res_blocks):
            blocks.append(ResidualBlock(base_channels))
        self.blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Conv2d(base_channels, latent_dim, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        squeeze = False

        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze = True

        B, C, H, W = z.shape
        assert C == self.latent_dim, f"Expected {self.latent_dim} channels, got {C}"

        if labels.dim() == 0:
            labels = labels.expand(B)
        labels = labels.long()

        e = self.label_embed(labels)
        e_map = e.view(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([z, e_map], dim=1)

        x = self.in_conv(x)
        x = F.silu(x)
        x = self.blocks(x)
        x = self.out_conv(x)

        if squeeze:
            x = x.squeeze(0)

        return x
