import torch
import torch.nn as nn

from utils.hardware.hardware_utils import select_device

from .unet_encoder_decoder import (
    UNetBottleneck, UNetDecoder, UNetEncoder
)
from .unet_blocks import StableDiffusionAttentionBlock


class UNet(nn.Module):
    """
    UNet model with encoder, bottleneck, and decoder components.

    - Encoder: Downsampling blocks with optional self-attention and cross-attention.
    - Bottleneck: Intermediate processing block.
    - Decoder: Upsampling blocks with optional self-attention and cross-attention.

    Args:
        model_config: Model configuration.
        device: Device to run the model on (e.g., 'mps', 'cuda', 'cpu').
    """

    # ===== Initialization =====
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        time_emb_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Initialize the model device
        self.device = select_device(device)

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

    # ===== Core Operations =====
    def forward(self, x, t):
        x, skips = self.encoder(x, t)
        x = self.bottleneck(x, t)
        x = self.decoder(x, skips, t)
        return x


class StableDiffusionUNet(nn.Module):
    """
    基于现有UNet的Stable Diffusion风格模型
    专门针对汉字生成任务优化，显存友好
    """

    # ===== Initialization =====
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 320,  # 基础通道数，比标准SD小
        time_emb_dim: int = 1280,  # 时间嵌入维度
        device: torch.device = None,
    ):
        super().__init__()
        
        # 初始化模型设备
        self.device = select_device(device)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        
        # 使用现有的UNet架构，但增加通道数
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            device=device
        )
        
        # 添加改进的注意力层到瓶颈部分
        self.attention_layers = nn.ModuleList([
            StableDiffusionAttentionBlock(
                channels=base_channels * 8,  # 瓶颈层通道数
                num_heads=8,
                use_checkpoint=False,
            ),
            # 添加第二个注意力层增强特征提取
            StableDiffusionAttentionBlock(
                channels=base_channels * 8,
                num_heads=8,
                use_checkpoint=False,
            )
        ])
        
        # 添加残差连接和归一化
        self.attention_norm = nn.GroupNorm(32, base_channels * 8)
        
        # 添加可学习缩放参数
        self.attention_scale = nn.Parameter(torch.ones(1))
        
        # 移动模型到指定设备
        self.to(self.device)
    
    def get_time_embedding(self, timesteps):
        """获取时间嵌入 - 使用与原有UNet相同的实现"""
        # 直接使用原有UNet的时间嵌入，不需要重新实现
        # 这里返回原始的时间步，让原有UNet的encoder/decoder处理
        return timesteps

    # ===== Core Operations =====
    def forward(self, x, t, y=None):
        """
        前向传播 - 改进的Stable Diffusion风格
        Args:
            x: 输入张量 [B, C, H, W]
            t: 时间步 [B]
            y: 条件信息（可选）
        """
        # 获取时间嵌入
        t_emb = self.get_time_embedding(t)
        
        # 使用原有UNet进行编码
        x, skips = self.unet.encoder(x, t_emb)
        
        # 保存瓶颈层输入用于残差连接
        bottleneck_input = x
        
        # 在瓶颈层应用改进的注意力机制
        x = self.attention_norm(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        
        # 添加残差连接和可学习缩放
        x = bottleneck_input + self.attention_scale * x
        
        # 使用原有UNet进行解码
        x = self.unet.decoder(x, skips, t_emb)
        
        return x
    
    def get_model_size(self):
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_M": total_params / 1e6,
            "trainable_params_M": trainable_params / 1e6,
        }
    
    def get_memory_usage(self, batch_size=1, image_size=(64, 64)):
        """估算显存使用量"""
        # 创建虚拟输入
        dummy_input = torch.randn(batch_size, self.in_channels, *image_size, device=self.device)
        dummy_timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        # 估算显存使用
        input_memory = dummy_input.numel() * dummy_input.element_size()
        
        # 模型参数显存
        model_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        
        # 梯度显存（训练时）
        gradient_memory = model_memory
        
        # 优化器状态显存（Adam）
        optimizer_memory = 2 * model_memory
        
        return {
            "input_memory_MB": input_memory / 1024 / 1024,
            "model_memory_MB": model_memory / 1024 / 1024,
            "gradient_memory_MB": gradient_memory / 1024 / 1024,
            "optimizer_memory_MB": optimizer_memory / 1024 / 1024,
            "total_training_memory_MB": (input_memory + model_memory + gradient_memory + optimizer_memory) / 1024 / 1024,
            "total_inference_memory_MB": (input_memory + model_memory) / 1024 / 1024,
        }
