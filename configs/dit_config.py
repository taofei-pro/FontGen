from dataclasses import dataclass


@dataclass
class DiTModelConfig:
    token_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class DiTTrainingConfig:
    batch_size: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 400
    warmup_epochs: int = 20
    guidance_scale: float = 3.0
