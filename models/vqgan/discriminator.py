from __future__ import annotations

import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator placeholder."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)
