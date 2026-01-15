from __future__ import annotations

import torch
import torch.nn as nn


class DiTModel(nn.Module):
    """Diffusion Transformer placeholder for token generation."""

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.net = nn.Identity()

    def forward(self, tokens: torch.Tensor, cond_tokens: torch.Tensor | None = None) -> torch.Tensor:
        _ = cond_tokens
        return self.net(tokens)

    def sample(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        return cond_tokens
