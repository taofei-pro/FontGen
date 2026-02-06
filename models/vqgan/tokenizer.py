from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantizer for VQGAN-2 with EMA updates for codebook."""

    def __init__(self, codebook_size: int, embed_dim: int, commitment_cost: float, decay: float = 0.99, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        
        # EMA parameters
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', torch.clone(self.embedding.weight))

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
        
        # EMA codebook update
        if self.training:
            encoding_one_hot = F.one_hot(encoding_indices, self.codebook_size).float()
            
            # Update EMA cluster size
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encoding_one_hot, dim=0)
            
            # Update EMA embedding weights
            dw = encoding_one_hot.t() @ flat_z
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
            
            # Normalize EMA weights
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            
            # Update embedding weights using EMA
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        return quantized_st, loss


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(x) + self.conv5(x))


class VQGAN2Tokenizer(nn.Module):
    """VQGAN-2 tokenizer (deeper + multi-scale encoder/decoder)."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 4,
        base_channels: int = 128,
        token_dim: int = 256,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        multiscale: bool = True,
        coarse_downsample: int = 2,
        coarse_weight: float = 0.5,
        tanh_output: bool = True,
        vq_decay: float = 0.99,
        vq_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.codebook_size = codebook_size
        self.use_multiscale = multiscale
        self.coarse_downsample = coarse_downsample
        self.coarse_weight = coarse_weight
        self.tanh_output = tanh_output
        self.enc_head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc_block1 = nn.Sequential(
            MultiScaleBlock(base_channels),
            ResidualBlock(base_channels),
        )
        self.enc_down1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.enc_block2 = nn.Sequential(
            MultiScaleBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
        )
        self.enc_down2 = nn.Conv2d(
            base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.enc_block3 = nn.Sequential(
            MultiScaleBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )
        self.enc_out = nn.Conv2d(base_channels * 4, token_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embed_dim=token_dim,
            commitment_cost=commitment_cost,
            decay=vq_decay,
            epsilon=vq_epsilon,
        )
        self.coarse_quantizer = None
        if self.use_multiscale:
            self.coarse_quantizer = VectorQuantizer(
                codebook_size=codebook_size,
                embed_dim=token_dim,
                commitment_cost=commitment_cost,
                decay=vq_decay,
                epsilon=vq_epsilon,
            )
        self.dec_in = nn.Conv2d(token_dim, base_channels * 4, kernel_size=1)
        self.dec_block3 = nn.Sequential(
            MultiScaleBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )
        self.dec_up1 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.dec_block2 = nn.Sequential(
            MultiScaleBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
        )
        self.dec_up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.dec_block1 = nn.Sequential(
            MultiScaleBlock(base_channels),
            ResidualBlock(base_channels),
        )
        self.dec_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.enc_head(images)
        x = self.enc_block1(x)
        x = self.enc_down1(x)
        x = self.enc_block2(x)
        x = self.enc_down2(x)
        x = self.enc_block3(x)
        z = self.enc_out(x)
        quant_fine, loss_fine = self.quantizer(z)
        if not self.use_multiscale or self.coarse_quantizer is None:
            return quant_fine, loss_fine

        if self.coarse_downsample <= 1:
            raise ValueError("coarse_downsample must be > 1 for multiscale tokenizer.")
        z_coarse = F.avg_pool2d(
            z,
            kernel_size=self.coarse_downsample,
            stride=self.coarse_downsample,
        )
        quant_coarse, loss_coarse = self.coarse_quantizer(z_coarse)
        quant_coarse_up = F.interpolate(
            quant_coarse, size=quant_fine.shape[-2:], mode="nearest"
        )
        return quant_fine + self.coarse_weight * quant_coarse_up, loss_fine + loss_coarse

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.dec_in(tokens)
        x = self.dec_block3(x)
        x = F.leaky_relu(self.dec_up1(x), negative_slope=0.2)
        x = self.dec_block2(x)
        x = F.leaky_relu(self.dec_up2(x), negative_slope=0.2)
        x = self.dec_block1(x)
        x = self.dec_out(x)
        x = torch.clamp(x, min=-5, max=5)
        
        if self.tanh_output:
            x = torch.tanh(x)  # [-1, 1]
        return x

    def downsample_factor(self, probe_size: int = 64) -> int:
        """Estimate spatial downsample factor of the encoder."""
        if probe_size <= 0:
            raise ValueError("probe_size must be positive.")
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, probe_size, probe_size, device=device)
            z = self.enc_out(self.enc_block3(self.enc_down2(self.enc_block2(self.enc_down1(self.enc_block1(self.enc_head(x)))))))
        factor = probe_size // z.shape[-1]
        if factor <= 0:
            raise ValueError("Invalid downsample factor computed.")
        return factor
