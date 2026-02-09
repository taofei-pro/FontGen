from dataclasses import dataclass


@dataclass
class DiTModelConfig:
    token_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    time_steps: int = 1000
    window_size: int = 8
    shift_window: bool = True


@dataclass
class DiTDatasetConfig:
    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"
    split_ratios: tuple[float, float] = (0.9, 0.1)
    random_seed: int = 2025
    batch_size: int = 1
    num_workers: int = 2


@dataclass
class DiTTrainingConfig:
    batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 800
    warmup_epochs: int = 20
    guidance_scale: float = 3.0
