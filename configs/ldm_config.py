from dataclasses import dataclass


@dataclass
class LDMModelConfig:
    # VQVAE 配置
    vqvae_in_channels: int = 1  # VQVAE 输入图像的通道数，1表示灰度图
    vqvae_base_channels: int = 64  # VQVAE 基础通道数
    vqvae_latent_dim: int = 32  # VQVAE 潜在空间的维度
    vqvae_codebook_size: int = 512  # VQVAE 码本大小
    vqvae_commitment_cost: float = 0.1  # VQVAE 承诺损失的权重
    
    # LDM 配置
    time_pos_dim: int = 256  # 时间位置编码的维度
    time_emb_dim: int = 1024  # 时间嵌入的维度
    time_steps: int = 1000  # 扩散模型的时间步数
    unet_base_channels: int = 64  # UNet 基础通道数


@dataclass
class LDMTrainingConfig:
    batch_size: int = 1  # 训练批量大小
    learning_rate: float = 1e-4  # 学习率
    num_epochs: int = 100  # 训练轮数
    save_dir: str = "checkpoints"  # 模型保存目录
    vqvae_checkpoint: str = "checkpoints/vqvae.pth"  # VQVAE 预训练权重路径
