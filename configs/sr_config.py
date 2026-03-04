from dataclasses import dataclass


@dataclass
class SRModelConfig:
    model_name: str = "realesrgan"  # 超分模型名称，如 "realesrgan", "swinir", "edsr"
    scale: int = 2  # 超分缩放因子


@dataclass
class SRTrainingConfig:
    batch_size: int = 2  # 训练批量大小
    learning_rate: float = 1e-4  # 学习率
    num_epochs: int = 200  # 训练轮数
