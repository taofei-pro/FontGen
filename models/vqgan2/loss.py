from __future__ import annotations

import torch
import torch.nn as nn


class VQGAN2Loss(nn.Module):
    """Composite loss placeholder for VQGAN-2."""

    def __init__(self) -> None:
        super().__init__()
        self.recon_loss = nn.L1Loss()

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.recon_loss(recon, target)
