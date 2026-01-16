from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EDSR(nn.Module):
    def __init__(self, scale: int = 2, channels: int = 64, num_blocks: int = 8) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1 * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        res = self.body(x)
        x = x + res
        return self.tail(x)


class TorchScriptSR(nn.Module):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        self.model = torch.jit.load(ckpt_path, map_location="cpu").eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SRModel(nn.Module):
    """Super-resolution model wrapper."""

    def __init__(
        self,
        scale: int = 2,
        model_name: str = "basic",
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.model_name = model_name
        self.is_torchscript = False
        if model_name in {"realesrgan", "swinir"}:
            if ckpt_path and ckpt_path.endswith((".pt", ".jit")):
                self.net = TorchScriptSR(ckpt_path)
                self.is_torchscript = True
            else:
                print(
                    "Warning: external SR model requires a TorchScript checkpoint; "
                    "falling back to EDSR."
                )
                self.model_name = "edsr"
                self.net = EDSR(scale=scale)
        elif model_name == "edsr":
            self.net = EDSR(scale=scale)
        else:
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1 * scale * scale, kernel_size=3, padding=1),
                nn.PixelShuffle(scale),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
