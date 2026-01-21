from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode structure condition maps into tokens."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        downsample_factor: int = 4,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if self.downsample_factor > 1:
            cond = nn.functional.avg_pool2d(
                cond, kernel_size=self.downsample_factor, stride=self.downsample_factor
            )
        return self.proj(cond)
