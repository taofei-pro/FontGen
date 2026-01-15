import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEDownBlock(nn.Module):
    """
    Enhanced down-sample block with residual connections and attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(max(out_channels // 8, 1), out_channels)
        self.norm2 = nn.GroupNorm(max(out_channels // 8, 1), out_channels)
        self.activation = nn.SiLU()
        
        # Residual connection (if channel dimensions match)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Attention mechanism (简化版本，避免通道数问题)
        if out_channels >= 4:
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, max(out_channels // 4, 1), 1),
                nn.SiLU(),
                nn.Conv2d(max(out_channels // 4, 1), out_channels, 1),
                nn.Sigmoid()
            )
        else:
            # 对于通道数太少的层，使用简单的注意力
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # Store input for residual connection
        residual = self.residual_conv(x)
        
        # Main path
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        
        # Apply lightweight attention (减少计算量)
        if x.shape[1] >= 4:  # 只对通道数>=4的特征应用注意力
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Residual connection (after downsampling)
        x = x + residual[:, :, ::2, ::2]  # Downsample residual to match spatial dimensions
        
        return x


class VQVAEUpBlock(nn.Module):
    """
    Enhanced up-sample block with residual connections and attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        # Main upsampling path
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(max(out_channels // 8, 1), out_channels)
        self.norm2 = nn.GroupNorm(max(out_channels // 8, 1), out_channels)
        self.activation = nn.SiLU()
        
        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Attention mechanism (简化版本，避免通道数问题)
        if out_channels >= 4:
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, max(out_channels // 4, 1), 1),
                nn.SiLU(),
                nn.Conv2d(max(out_channels // 4, 1), out_channels, 1),
                nn.Sigmoid()
            )
        else:
            # 对于通道数太少的层，使用简单的注意力
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # Store input for residual connection
        residual = self.residual_conv(x)
        
        # Main path
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        
        # Apply lightweight attention (减少计算量)
        if x.shape[1] >= 4:  # 只对通道数>=4的特征应用注意力
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Residual connection (upsample residual to match spatial dimensions)
        residual_upsampled = F.interpolate(residual, size=x.shape[2:], mode='nearest')
        x = x + residual_upsampled
        
        return x


class VQVAEOutBlock(nn.Module):
    """
    Output block for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x
