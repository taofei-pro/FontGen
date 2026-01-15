from dataclasses import dataclass


@dataclass
class VQGAN2ModelConfig:
    input_img_channels: int = 1
    base_channels: int = 128
    latent_dim: int = 4
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
