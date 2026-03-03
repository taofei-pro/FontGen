import torch
import torch.nn as nn


class TimeResidual(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        residual = x

        x = self.conv_block1(x)
        time = self.time_emb(t)
        x = x + time.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_block2(x)

        return x + residual


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, downsample: bool = True):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )
        self.time_res_block = TimeResidual(
            in_channels=out_channels,
            time_emb_dim=time_emb_dim,
        )

        self.down_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        ) if downsample else nn.Identity()

    def forward(self, x, t):
        x = self.conv_block(x)
        x = self.time_res_block(x, t)

        skip = x
        x = self.down_block(x)
        return x, skip


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, upsample: bool = True):
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        ) if upsample else nn.Identity()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )
        self.time_res_block = TimeResidual(
            in_channels=out_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x, skip, t):
        x = self.up_block(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        x = self.time_res_block(x, t)

        return x


class UNetOutBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        )

        self.time_res_block = TimeResidual(
            in_channels=in_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x, t):
        x = self.conv_block(x)
        x = self.time_res_block(x, t)

        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, time_emb_dim: int):
        super().__init__()

        # Adjusted for 8x8 input
        ch_multipliers = [1, 2, 4]
        channels = [in_channels] + [base_channels * m for m in ch_multipliers]

        self.encoder_blocks = nn.ModuleList(
            [
                UNetDownBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    time_emb_dim=time_emb_dim,
                    downsample=False if i == 0 else True,
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, t):
        skips = []
        for block in self.encoder_blocks:
            x, skip = block(x, t)
            skips.append(skip)
        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, out_channels: int, base_channels: int, time_emb_dim: int):
        super().__init__()

        # Adjusted for 8x8 output
        ch_multipliers = [4, 2, 1]
        channels = [base_channels * m for m in ch_multipliers] + [base_channels]

        self.decoder_blocks = nn.ModuleList(
            [
                UNetUpBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    time_emb_dim=time_emb_dim,
                    upsample=False if i == len(channels) - 2 else True,
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.output_block = UNetOutBlock(
            in_channels=channels[-1],
            out_channels=out_channels,
        )

    def forward(self, x, skips, t):
        for block in self.decoder_blocks:
            x = block(x, skips.pop(), t)
        x = self.output_block(x)
        return x


class UNetBottleneck(nn.Module):
    def __init__(self, base_channels: int, time_emb_dim: int):
        super().__init__()

        channels = base_channels * 4

        self.bottleneck_blocks = nn.ModuleList(
            [
                BottleneckBlock(
                    in_channels=channels,
                    time_emb_dim=time_emb_dim,
                )
                for _ in range(3)
            ]
        )

    def forward(self, x, t):
        for block in self.bottleneck_blocks:
            x = block(x, t)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int, time_emb_dim: int, device: torch.device):
        super().__init__()
        self.device = device

        # Define the encoder, bottleneck, and decoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )
        self.bottleneck = UNetBottleneck(
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )
        self.decoder = UNetDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )

        # Move model to the specified device
        self.to(self.device)

    def forward(self, x, t):
        x, skips = self.encoder(x, t)
        x = self.bottleneck(x, t)
        x = self.decoder(x, skips, t)
        return x