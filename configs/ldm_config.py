from dataclasses import dataclass


@dataclass
class LDMModelConfig:
    # LDM 配置
    time_pos_dim: int = 256  # 时间位置编码的维度
    time_emb_dim: int = 1024  # 时间嵌入的维度
    time_steps: int = 1000  # 扩散模型的时间步数
    unet_base_channels: int = 128  # UNet 基础通道数 (increased for better capacity)


@dataclass
class LDMTrainingConfig:
    batch_size: int = 8  # 训练批量大小 (reduced to fit larger model)
    learning_rate: float = 5e-4  # 学习率
    num_epochs: int = 350  # 训练轮数 (increased for better convergence)
    save_dir: str = "checkpoints"  # 模型保存目录
    vqvae_checkpoint: str = "checkpoints/vqvae.pth"  # VQVAE 预训练权重路径
