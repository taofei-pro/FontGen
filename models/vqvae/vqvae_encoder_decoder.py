import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vqvae_blocks import VQVAEDownBlock, VQVAEOutBlock, VQVAEUpBlock


class VQVAEEncoder(nn.Module):
    """
    Enhanced Encoder with skip connections for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
    ):
        super().__init__()
        ch_multipliers = [1, 2, 4]
        channels = [in_channels] + [base_channels * m for m in ch_multipliers]

        self.encoder_blocks = nn.ModuleList(
            [
                VQVAEDownBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.encoder_blocks.append(
            nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
        )
        
        # 存储跳跃连接特征
        self.skip_features = []

    def forward(self, x):
        self.skip_features = []  # 重置跳跃连接
        for i, block in enumerate(self.encoder_blocks):
            if i < len(self.encoder_blocks) - 1:  # 不是最后一层
                self.skip_features.append(x)  # 保存跳跃连接特征
            x = block(x)
        return x


class VQVAEDecoder(nn.Module):
    """
    Enhanced Decoder with skip connections for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
    ):
        super().__init__()

        ch_multipliers = [4, 2, 1]
        channels = [base_channels * m for m in ch_multipliers] + [out_channels]

        self.decoder_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]
        )
        self.decoder_blocks.extend(
            [
                VQVAEUpBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.decoder_blocks.append(
            VQVAEOutBlock(
                in_channels=channels[-1],
                out_channels=out_channels,
            )
        )
        
        # 简化跳跃连接：只在相同分辨率层之间连接，不做通道转换
        # 这样可以避免复杂的通道匹配逻辑，减少显存占用
        # 编码器输出特征顺序（从输入到潜在）：[1ch@64x64, 128ch@32x32, 256ch@16x16]
        # 解码器输入特征顺序（从潜在到输出）：[512ch@8x8, 256ch@16x16, 128ch@32x32, 1ch@64x64]
        # 我们将跳跃连接简化为可选功能，避免显存问题
        self.use_skip_connections = False  # 暂时禁用跳跃连接，避免显存问题

    def forward(self, x, skip_features=None):
        for i, block in enumerate(self.decoder_blocks):
            # 暂时禁用跳跃连接，避免显存和通道匹配问题
            # 如果需要启用，需要正确实现通道匹配逻辑
            x = block(x)
        return x


class VQVAEQuantizer(nn.Module):
    """
    Vector Quantizer for the VQ-VAE model.
    """

    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        commitment_cost: float,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost

        self.emb = nn.Embedding(codebook_size, latent_dim)
        self.emb.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x):
        x_perm = rearrange(x, "b c h w -> b h w c")
        b, h, w, c = x_perm.shape

        flat_x = rearrange(x_perm, "b h w c -> (b h w) c")

        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.emb.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.emb.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        quantized = self.emb(encoding_indices.squeeze(1))
        quantized = rearrange(quantized, "(b h w) c -> b h w c", b=b, h=h, w=w)

        e_latent_loss = F.mse_loss(quantized.detach(), x_perm)
        q_latent_loss = F.mse_loss(quantized, x_perm.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x_perm + (quantized - x_perm).detach()

        quantized = rearrange(quantized, "b h w c -> b c h w")

        return quantized, loss
