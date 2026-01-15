import torch
import torch.nn as nn

from .unet_blocks import (
    BottleneckBlock, UNetDownBlock, UNetOutBlock, UNetUpBlock,
    StableDiffusionResBlock, StableDiffusionAttentionBlock,
    StableDiffusionDownsample, StableDiffusionUpsample
)


class UNetEncoder(nn.Module):
    """
    Encoder for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        ch_multipliers = [1, 2, 4, 8]
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
    """
    Decoder for the UNet model.
    """

    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        ch_multipliers = [8, 4, 2, 1]
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
    """
    Bottleneck for the UNet model.
    """

    def __init__(
        self,
        base_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        channels = base_channels * 8

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


class StableDiffusionUNetEncoder(nn.Module):
    """
    Stable Diffusion风格的UNet编码器
    专门针对汉字生成任务优化
    """
    
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (4, 2, 1),
        dropout: float = 0.0,
        channel_mult: tuple = (1, 2, 4, 4),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 8,
        num_head_channels: int = -1,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels_mult = out_channels_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_new_attention_order = use_new_attention_order
        
        # 输入投影
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # 下采样块
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    StableDiffusionResBlock(
                        ch,
                        model_channels * 4,  # time_emb_dim
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        StableDiffusionAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                ds *= 2
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    StableDiffusionDownsample(ch, conv_resample, dims=dims)
                )
                ds *= 2
    
    def forward(self, x, emb):
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, StableDiffusionResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, StableDiffusionAttentionBlock):
                        h = layer(h)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        return h, hs


class StableDiffusionUNetDecoder(nn.Module):
    """
    Stable Diffusion风格的UNet解码器
    专门针对汉字生成任务优化
    """
    
    def __init__(
        self,
        out_channels: int,
        model_channels: int,
        out_channels_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (4, 2, 1),
        dropout: float = 0.0,
        channel_mult: tuple = (1, 2, 4, 4),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 8,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.out_channels_mult = out_channels_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_new_attention_order = use_new_attention_order
        
        # 上采样块
        self.output_blocks = nn.ModuleList([])
        ch = model_channels * channel_mult[-1]  # 从最大通道数开始
        ds = 1
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = ch + (mult * model_channels if i == 0 else 0)
                layers = [
                    StableDiffusionResBlock(
                        ich,
                        model_channels * 4,  # time_emb_dim
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        StableDiffusionAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample if num_heads_upsample != -1 else num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        StableDiffusionUpsample(ch, conv_resample, dims=dims)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, hs, emb):
        for module in self.output_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, StableDiffusionResBlock):
                        x = layer(x, emb)
                    elif isinstance(layer, StableDiffusionAttentionBlock):
                        x = layer(x)
                    elif isinstance(layer, StableDiffusionUpsample):
                        x = layer(x)
                    else:
                        x = layer(x)
            else:
                x = module(x)
        
        return self.out(x)


class StableDiffusionUNetBottleneck(nn.Module):
    """
    Stable Diffusion风格的UNet瓶颈层
    专门针对汉字生成任务优化
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 8,
        num_head_channels: int = -1,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        
        self.channels = channels
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_new_attention_order = use_new_attention_order
        
        self.middle_block = nn.Sequential(
            StableDiffusionResBlock(
                channels,
                time_emb_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            StableDiffusionAttentionBlock(
                channels,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            StableDiffusionResBlock(
                channels,
                time_emb_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
    
    def forward(self, x, emb):
        for module in self.middle_block:
            if isinstance(module, StableDiffusionResBlock):
                x = module(x, emb)
            elif isinstance(module, StableDiffusionAttentionBlock):
                x = module(x)
            else:
                x = module(x)
        return x
