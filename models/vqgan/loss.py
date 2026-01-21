from __future__ import annotations

import torch
import torch.nn as nn


class VQGAN2Loss(nn.Module):
    """Composite loss with L1 + perceptual (LPIPS)."""

    def __init__(self, perceptual_weight: float = 0.4) -> None:
        super().__init__()
        self.recon_loss = nn.L1Loss()
        self.perceptual_weight = perceptual_weight
        self._lpips = None
        try:
            import lpips  # type: ignore

            self._lpips = lpips.LPIPS(net="vgg").eval()
        except Exception:
            self._lpips = None
            self.perceptual_weight = 0.0

    def _prepare_lpips_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def compute_losses(self, recon: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l1 = self.recon_loss(recon, target)
        perceptual = torch.zeros_like(l1)
        if self._lpips is not None and self.perceptual_weight > 0:
            recon_lp = self._prepare_lpips_input(recon)
            target_lp = self._prepare_lpips_input(target)
            perceptual = self._lpips(recon_lp, target_lp).mean()
        total = l1 + self.perceptual_weight * perceptual
        return {
            "l1": l1,
            "perceptual": perceptual,
            "total": total,
        }

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute_losses(recon, target)["total"]
