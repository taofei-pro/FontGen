from dataclasses import dataclass


@dataclass
class LDMDatasetConfig:
    """
    Configuration class for the LDM dataset settings.
    """

    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"

    splits_root: str = "charsets"
    split_ratios: tuple[float, float] = (0.9, 0.1)
    random_seed: int = 2025
    batch_size: int = 4          # 降低批次大小，避免显存不足（OOM优化）(8 → 4)
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
class LDMModelConfig:
    """
    Configuration class for the LDM architecture settings.
    """

    unet_base_channels: int = 96     # 回退到稳定配置 (112 → 96)

    time_pos_dim: int = 256          # 保持256
    time_emb_dim: int = 1792         # 回退到稳定配置 (2048 → 1792)
    time_steps: int = 1400           # 回退到稳定配置 (1600 → 1400)
    
    # 默认使用Stable Diffusion架构
    use_stable_diffusion: bool = True


@dataclass
class StableDiffusionLDMModelConfig:
    """
    Configuration class for the Stable Diffusion LDM architecture settings.
    """

    # UNet架构参数
    model_channels: int = 320        # 回退到稳定配置 (384 → 320)
    out_channels_mult: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple = (4, 2, 1)
    dropout: float = 0.0
    channel_mult: tuple = (1, 2, 4, 4)
    conv_resample: bool = True
    dims: int = 2
    num_classes: int = None
    use_checkpoint: bool = False
    use_fp16: bool = False
    num_heads: int = 8
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    use_spatial_transformer: bool = False
    transformer_depth: int = 1
    context_dim: int = None
    n_embed: int = None
    legacy: bool = True

    # 时间嵌入参数
    time_pos_dim: int = 256
    time_emb_dim: int = 1280         # model_channels * 4 (320 * 4)
    time_steps: int = 1000           # 回退到稳定配置 (1200 → 1000)

    # 噪声调度参数
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"


@dataclass
class LDMTrainingConfig:
    """
    Configuration class for the LDM training settings.
    """

    # 针对大数据集（9,497字符）优化
    learning_rate: float = 3e-4        # 降低学习率，提高稳定性 (4e-4 → 3e-4)
    min_learning_rate: float = 1e-6    # 保持默认值
    num_epochs: int = 450              # 减少训练轮数（数据量增加12.7倍）(600 → 450)
    warmup_epochs: int = 60            # 增加预热轮数，让模型稳定启动 (50 → 60)
    early_stopping_patience: int = 120  # 增加早停耐心，大数据集需要更多耐心 (100 → 120)

    pretrained_vqvae_path: str = "checkpoints/vqvae.pth"
    model_save_path: str = "checkpoints/ldm.pth"

    tensorboard_log_dir: str = "runs/LDM"

    sample_root: str = "samples"
    train_split: str = "train"
    val_split: str = "val"
    gt_split: str = "eval_outputs/gt"
    gen_split: str = "eval_outputs/gen"

    sample_steps: int = 200            # 进一步增加采样步数，提升生成质量

    img_save_interval: int = 5
    lpips_eval_interval: int = 10
    eval_batch_size: int = 2


@dataclass
class StableDiffusionLDMTrainingConfig:
    """
    Configuration class for the Stable Diffusion LDM training settings.
    """

    # 针对大数据集（9,497字符）优化
    learning_rate: float = 4e-4        # 降低学习率，提高稳定性 (5e-4 → 4e-4)
    min_learning_rate: float = 1e-6    # 保持默认值
    num_epochs: int = 600              # 减少训练轮数（数据量增加12.7倍）(800 → 600)
    warmup_epochs: int = 120           # 增加预热轮数，让模型稳定启动 (100 → 120)
    early_stopping_patience: int = 120  # 增加早停耐心，大数据集需要更多耐心 (100 → 120)

    pretrained_vqvae_path: str = "checkpoints/vqvae.pth"
    model_save_path: str = "checkpoints/stable_diffusion_ldm.pth"

    tensorboard_log_dir: str = "runs/StableDiffusionLDM"

    sample_root: str = "samples_stable_diffusion"
    train_split: str = "train"
    val_split: str = "val"
    gt_split: str = "eval_outputs/gt"
    gen_split: str = "eval_outputs/gen"

    sample_steps: int = 250            # 进一步增加采样步数，提升生成质量

    img_save_interval: int = 5
    lpips_eval_interval: int = 10
    eval_batch_size: int = 2


@dataclass
class LDMInferenceConfig:
    """
    Configuration class for the LDM inference settings.
    """

    pretrained_ldm_path: str = "checkpoints/ldm.pth"

    sample_root: str = "samples"
    ref_split: str = "inference/ref"
    gt_split: str = "inference/gt"
    gen_split: str = "inference/gen"

    batch_size: int = 16
    sample_steps: int = 50


@dataclass
class StableDiffusionLDMInferenceConfig:
    """
    Configuration class for the Stable Diffusion LDM inference settings.
    """

    pretrained_ldm_path: str = "checkpoints/stable_diffusion_ldm.pth"

    sample_root: str = "samples_stable_diffusion"
    ref_split: str = "inference/ref"
    gt_split: str = "inference/gt"
    gen_split: str = "inference/gen"

    batch_size: int = 16
    sample_steps: int = 150            # 增加采样步数，提升生成质量
