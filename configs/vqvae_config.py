from dataclasses import dataclass


@dataclass
class VQVAEDatasetConfig:
    """
    Configuration class for the VQVAE dataset settings.
    """

    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"

    splits_root: str = "charsets"
    split_ratios: tuple[float, float] = (0.9, 0.1)
    random_seed: int = 2025
    batch_size: int = 4          # 降低批次大小，避免显存不足（OOM优化）(6 → 4)
    num_workers: int = 2         # 减少数据加载worker，降低显存占用（OOM优化）(4 → 2)
    
    # 数据增强配置（针对字形数据优化）
    use_data_augmentation: bool = False   # 禁用数据增强 - 字形数据对几何变换过于敏感
    augmentation_type: str = "basic"       # 增强类型（已禁用）
    rotation_range: float = 1.0            # 旋转角度范围（度）- 已禁用
    scale_range: tuple[float, float] = (0.98, 1.02)  # 缩放范围 - 已禁用
    noise_std: float = 0.002               # 噪声标准差 - 已禁用
    brightness_range: tuple[float, float] = (0.95, 1.05)  # 亮度调整范围 - 已禁用
    contrast_range: tuple[float, float] = (0.95, 1.05)    # 对比度调整范围 - 已禁用
    augmentation_prob: float = 0.0        # 应用增强的概率 - 已禁用


@dataclass
class VQVAEModelConfig:
    """
    Configuration class for the VQVAE architecture settings.
    """

    input_img_channels: int = 1
    encoder_base_channels: int = 96  # 平衡性能和速度 (128 → 96, 原稳定值)
    latent_dim: int = 4              # 平衡表达能力和效率 (6 → 4, 原稳定值)
    codebook_size: int = 192         # 充足的码本容量 (256 → 192)
    commitment_cost: float = 0.25
    
    # VQ-GAN settings (默认启用VQ-GAN模式)
    use_vqgan: bool = True           # 是否启用VQ-GAN模式 (默认启用)
    discriminator_lr: float = 2e-4   # 降低判别器学习率，提高稳定性 (3e-4 → 2e-4)
    perceptual_weight: float = 0.4   # 增加感知损失权重，提高质量 (0.3 → 0.4)
    adversarial_weight: float = 0.1  # 降低对抗损失权重，减少训练不稳定 (0.15 → 0.1)


@dataclass
class VQVAETrainingConfig:
    """
    Configuration class for VQ-VAE training settings.
    """

    # 针对大数据集（9,497字符）优化
    learning_rate: float = 6e-4        # 降低学习率，提高稳定性 (8e-4 → 6e-4)
    min_learning_rate: float = 1e-6    # 保持默认值
    num_epochs: int = 300              # 减少训练轮数（数据量增加12.7倍，每个epoch数据更多）(400 → 300)
    warmup_epochs: int = 40            # 增加预热轮数，让模型稳定启动 (30 → 40)
    early_stopping_patience: int = 120  # 增加早停耐心，大数据集需要更多耐心 (100 → 120)
    
    # 新增正则化参数
    weight_decay: float = 2e-4         # 增加权重衰减，防止过拟合
    dropout_rate: float = 0.1          # 添加dropout，提高泛化能力

    model_save_path: str = "checkpoints/vqvae.pth"
    tensorboard_log_dir: str = "runs/VQVAE"
