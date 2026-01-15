from __future__ import annotations

import torch


class DiffusionScheduler:
    """Basic linear beta scheduler for diffusion training."""

    def __init__(
        self,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ) -> None:
        self.steps = steps
        self.device = device or torch.device("cpu")
        betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alphas_cumprod) * x + torch.sqrt(1 - alphas_cumprod) * noise
