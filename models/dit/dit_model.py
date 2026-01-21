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
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
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

    def _window_partition(
        self, x: torch.Tensor, window_size: int, hw: tuple[int, int], shift_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        h, w = hw
        x = x.view(b, h, w, c)
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        x = (
            x.view(b, h // window_size, window_size, w // window_size, window_size, c)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        return x.view(-1, window_size * window_size, c), x

    def _window_reverse(
        self,
        x: torch.Tensor,
        window_size: int,
        hw: tuple[int, int],
        batch_size: int,
        shift_size: int,
    ) -> torch.Tensor:
        h, w = hw
        x = x.view(
            batch_size, h // window_size, w // window_size, window_size, window_size, -1
        )
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(batch_size, h, w, -1)
        )
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        return x.view(batch_size, h * w, -1)

    def forward(
        self,
        x: torch.Tensor,
        cond_seq: torch.Tensor | None = None,
        window_size: int | None = None,
        hw: tuple[int, int] | None = None,
        shift_size: int = 0,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        if window_size and hw and hw[0] % window_size == 0 and hw[1] % window_size == 0:
            x_win, _ = self._window_partition(x_norm, window_size, hw, shift_size)
            attn_out, _ = self.attn(x_win, x_win, x_win)
            attn_out = self._window_reverse(
                attn_out, window_size, hw, x.shape[0], shift_size
            )
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        if cond_seq is not None:
            cross_out, _ = self.cross_attn(self.norm_cross(x), cond_seq, cond_seq)
            x = x + cross_out
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
        window_size: int = 0,
        shift_window: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.time_steps = time_steps
        self.window_size = window_size
        self.shift_window = shift_window
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
        cond_seq: torch.Tensor | None = None,
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

        for idx, block in enumerate(self.blocks):
            shift_size = 0
            if self.window_size and self.shift_window and (idx % 2 == 1):
                shift_size = self.window_size // 2
            x = block(
                x,
                cond_seq=cond_seq,
                window_size=self.window_size,
                hw=(h, w),
                shift_size=shift_size,
            )

        x = self.out(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x
