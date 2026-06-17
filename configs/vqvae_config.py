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
    base_channels: int = 128  # 模型基础通道数 (优化: 64→128，增强特征提取能力)
    latent_dim: int = 64  # 潜在空间的维度 (优化: 2→64，提高重建质量)
    codebook_size: int = 512  # 码本大小 (优化: 64→512，增加离散表示能力)
    commitment_cost: float = 0.25  # 承诺损失的权重


@dataclass
class VQVAETrainingConfig:
    batch_size: int = 8  # 训练批量大小
    learning_rate: float = 5e-4  # 学习率 (优化: 1e-3→5e-4，提高稳定性)
    num_epochs: int = 200  # 训练轮数 (优化: 100→200，更充分训练)
    save_dir: str = "checkpoints"  # 模型保存目录
