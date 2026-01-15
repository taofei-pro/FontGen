from __future__ import annotations

import torch
import torch.nn as nn


class SRModel(nn.Module):
    """Super-resolution model wrapper placeholder."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
