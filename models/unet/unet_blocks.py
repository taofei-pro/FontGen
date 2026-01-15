import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeResidual(nn.Module):
    """
    Time residual block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
    ):
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
    """
    Downsample block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        downsample: bool = True
    ):
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
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=4, stride=2, padding=1),
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
    """
    Upsample block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        upsample: bool = True
    ):
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=4, stride=2, padding=1),
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
    """
    Output block for the UNet model.
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
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
    ):
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


class StableDiffusionAttentionBlock(nn.Module):
    """
    Stable Diffusion风格的注意力块
    专门针对汉字生成任务优化
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_checkpoint = use_checkpoint
        self.use_new_attention_order = use_new_attention_order
        
        if num_head_channels == -1:
            self.num_head_channels = channels // num_heads
        else:
            self.num_head_channels = num_head_channels
        
        # 确保通道数能被头数整除
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为注意力格式
        q = q.view(b, self.num_heads, self.num_head_channels, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, self.num_head_channels, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, self.num_head_channels, h * w).transpose(2, 3)
        
        # 计算注意力
        scale = 1.0 / math.sqrt(self.num_head_channels)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        # 重塑回原始格式
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return x + self.proj_out(out)


class StableDiffusionResBlock(nn.Module):
    """
    Stable Diffusion风格的残差块
    专门针对汉字生成任务优化
    """
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: int = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
    
    def forward(self, x, emb):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)
    
    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
            h = out_rest(h)
        else:
            h = h + emb_out.unsqueeze(-1).unsqueeze(-1)
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class StableDiffusionDownsample(nn.Module):
    """
    Stable Diffusion风格的下采样层
    """
    
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)
    
    def forward(self, x):
        return self.op(x)


class StableDiffusionUpsample(nn.Module):
    """
    Stable Diffusion风格的上采样层
    """
    
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
