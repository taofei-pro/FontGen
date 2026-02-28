from dataclasses import dataclass


@dataclass
class VQGAN2DatasetConfig:
    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"
    split_ratios: tuple[float, float] = (0.9, 0.1)
    random_seed: int = 2025
    batch_size: int = 1
    num_workers: int = 2


@dataclass
class VQGAN2ModelConfig:
    input_img_channels: int = 1
    base_channels: int = 64
    latent_dim: int = 4
    token_dim: int = 256
    codebook_size: int = 256
    commitment_cost: float = 0.1
    multiscale: bool = False
    coarse_downsample: int = 2
    coarse_weight: float = 0.5
    tanh_output: bool = True
    use_vqgan: bool = True
    vq_decay: float = 0.99
    vq_epsilon: float = 1e-5


@dataclass
class VQGAN2TrainingConfig:
    batch_size: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 300
    warmup_epochs: int = 30
    weight_decay: float = 1e-4
    discriminator_lr: float = 1e-4
    perceptual_weight: float = 0.3
    foreground_weight: float = 2.0
    adversarial_weight: float = 0.05
    discriminator_start_steps: int = 2000
    model_save_path: str = "checkpoints/vqgan.pth"