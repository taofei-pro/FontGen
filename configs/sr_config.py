from dataclasses import dataclass


@dataclass
class SRModelConfig:
    model_name: str = "realesrgan"
    scale: int = 2


@dataclass
class SRTrainingConfig:
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 200
