from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Basic vector quantizer for VQGAN-2."""

    def __init__(self, codebook_size: int, embed_dim: int, commitment_cost: float) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [B, C, H, W] -> [B*H*W, C]
        z_permute = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_permute.view(-1, self.embed_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(z_permute.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        loss = F.mse_loss(quantized.detach(), z) + self.commitment_cost * F.mse_loss(
            quantized, z.detach()
        )
        return quantized_st, loss


class VQGAN2Tokenizer(nn.Module):
    """VQGAN-2 tokenizer placeholder (encoder/decoder)."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 4,
        token_dim: int = 256,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.codebook_size = codebook_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, token_dim, kernel_size=1),
        )
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embed_dim=token_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(token_dim, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                latent_dim, latent_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                latent_dim, in_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(images)
        return self.quantizer(z)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens)

    def downsample_factor(self, probe_size: int = 64) -> int:
        """Estimate spatial downsample factor of the encoder."""
        if probe_size <= 0:
            raise ValueError("probe_size must be positive.")
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, self.encoder[0].in_channels, probe_size, probe_size, device=device)
            z = self.encoder(x)
        factor = probe_size // z.shape[-1]
        if factor <= 0:
            raise ValueError("Invalid downsample factor computed.")
        return factor
