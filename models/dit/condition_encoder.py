from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode structure condition maps into tokens (multi-scale)."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        downsample_factor: int = 4,
        use_multiscale: bool = True,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.use_multiscale = use_multiscale
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.downsample_factor > 1:
            cond_base = nn.functional.avg_pool2d(
                cond, kernel_size=self.downsample_factor, stride=self.downsample_factor
            )
        else:
            cond_base = cond
        cond_base = self.proj(cond_base)

        seqs = [cond_base.flatten(2).transpose(1, 2)]
        if self.use_multiscale:
            cond_coarse = nn.functional.avg_pool2d(cond_base, kernel_size=2, stride=2)
            seqs.append(cond_coarse.flatten(2).transpose(1, 2))

        cond_seq = torch.cat(seqs, dim=1)
        return cond_base, cond_seq
