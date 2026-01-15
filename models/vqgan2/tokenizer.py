from __future__ import annotations

import torch
import torch.nn as nn


class VQGAN2Tokenizer(nn.Module):
    """VQGAN-2 tokenizer placeholder (encoder/decoder)."""

    def __init__(self, latent_dim: int = 4, codebook_size: int = 512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens)
