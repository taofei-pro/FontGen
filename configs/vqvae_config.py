from dataclasses import dataclass


@dataclass
class VQVAEDatasetConfig:
    target_img_dir: str = "data/target"  # 目标字体图像目录
    reference_img_dir: str = "data/reference"  # 参考字体图像目录
    split_ratios: tuple[float, float] = (0.9, 0.1)  # 训练集和验证集的分割比例
    random_seed: int = 2025  # 随机种子，用于数据分割
    batch_size: int = 1  # 批量大小
    num_workers: int = 2  # 数据加载的工作线程数


@dataclass
class VQVAEModelConfig:
    in_channels: int = 1  # 输入图像的通道数，1表示灰度图
    base_channels: int = 64  # 模型基础通道数
    latent_dim: int = 2  # 潜在空间的维度
    codebook_size: int = 64  # 码本大小，决定了离散表示的词汇量
    commitment_cost: float = 0.25  # 承诺损失的权重


@dataclass
class VQVAETrainingConfig:
    batch_size: int = 8  # 训练批量大小
    learning_rate: float = 1e-3  # 学习率
    num_epochs: int = 100  # 训练轮数
    save_dir: str = "checkpoints"  # 模型保存目录
