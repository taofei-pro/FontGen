from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("token_dim must be divisible by 4 for 2D sin/cos embed.")
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device) / (dim // 4)
    omega = 1.0 / (10000 ** omega)
    out_y = torch.einsum("hw,d->hwd", y.float(), omega)
    out_x = torch.einsum("hw,d->hwd", x.float(), omega)
    pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
    return pos.view(1, height * width, dim)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DiTModel(nn.Module):
    """Transformer-based diffusion model for token denoising."""

    def __init__(
        self,
        token_dim: int = 256,
        time_steps: int = 1000,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.time_steps = time_steps
        self.time_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
        )

    def _time_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.token_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device).float() / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.token_dim:
            emb = torch.nn.functional.pad(emb, (0, self.token_dim - emb.shape[-1]))
        return self.time_mlp(emb)

    def forward(
        self,
        tokens: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond_tokens is None:
            cond_tokens = torch.zeros_like(tokens)
        x = tokens + cond_tokens
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        if timesteps is not None:
            t_emb = self._time_embed(timesteps).unsqueeze(1)
            x = x + t_emb

        pos = build_2d_sincos_pos_embed(h, w, self.token_dim, x.device)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.out(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x
