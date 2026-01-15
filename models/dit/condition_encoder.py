from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode structure condition maps into tokens."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(cond)
