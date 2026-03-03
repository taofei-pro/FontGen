import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VQVAEDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.down_block(x)


class VQVAEUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.up_block(x)


class VQVAEOutBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.conv_block(x)


class VQVAEQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float):
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


class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, out_channels: int):
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

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


class VQVAEDecoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, out_channels: int):
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

    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, latent_dim: int = 32, codebook_size: int = 512, commitment_cost: float = 0.25):
        super().__init__()
        self.encoder = VQVAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=latent_dim,
        )
        self.vector_quantizer = VQVAEQuantizer(
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = VQVAEDecoder(
            in_channels=latent_dim,
            base_channels=base_channels,
            out_channels=in_channels,
        )

    def forward(self, x):
        encoded_features = self.encoder(x)
        quantized_features, vq_loss = self.vector_quantizer(encoded_features)
        x_recon = self.decoder(quantized_features)
        return x_recon, vq_loss

    def encode(self, x):
        encoded_features = self.encoder(x)
        quantized_features, _ = self.vector_quantizer(encoded_features)
        return quantized_features

    def decode(self, x):
        return self.decoder(x)