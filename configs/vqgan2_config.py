from dataclasses import dataclass


@dataclass
class VQGAN2DatasetConfig:
    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"
    split_ratios: tuple[float, float] = (0.9, 0.1)
    random_seed: int = 2025
    batch_size: int = 4
    num_workers: int = 2


@dataclass
class VQGAN2ModelConfig:
    input_img_channels: int = 1
    base_channels: int = 128
    latent_dim: int = 4
    token_dim: int = 256
    codebook_size: int = 512
    commitment_cost: float = 0.25
    use_vqgan: bool = True


@dataclass
class VQGAN2TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 4e-4
    num_epochs: int = 300
    warmup_epochs: int = 20
    weight_decay: float = 1e-4
    discriminator_lr: float = 2e-4
    perceptual_weight: float = 0.4
    adversarial_weight: float = 0.1
    model_save_path: str = "checkpoints/vqgan2.pth"