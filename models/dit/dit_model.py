from __future__ import annotations

import torch
import torch.nn as nn


class DiTModel(nn.Module):
    """Minimal diffusion model head for token denoising."""

    def __init__(self, token_dim: int = 256, time_steps: int = 1000) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.time_embed = nn.Embedding(time_steps, token_dim)
        self.net = nn.Sequential(
            nn.Conv2d(token_dim * 2, token_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond_tokens is None:
            cond_tokens = torch.zeros_like(tokens)
        if timesteps is not None:
            t_emb = self.time_embed(timesteps).view(-1, self.token_dim, 1, 1)
            tokens = tokens + t_emb
        x = torch.cat([tokens, cond_tokens], dim=1)
        return self.net(x)
