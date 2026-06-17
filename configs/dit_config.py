from dataclasses import dataclass


@dataclass
class DiTModelConfig:
    token_dim: int = 256  # Token的维度
    num_layers: int = 12  # Transformer层数 (优化: 8→12，增强模型容量)
    num_heads: int = 8  # 注意力头数
    mlp_ratio: float = 4.0  # MLP隐藏层维度与输入维度的比率
    dropout: float = 0.05  # Dropout概率 (优化: 0.1→0.05，减少正则化)
    time_steps: int = 1000  # 扩散模型的时间步数
    window_size: int = 4  # 窗口注意力的窗口大小
    shift_window: bool = True  # 是否使用移位窗口注意力


@dataclass
class DiTDatasetConfig:
    target_img_dir: str = "data/target"  # 目标字体图像目录
    reference_img_dir: str = "data/reference"  # 参考字体图像目录
    split_ratios: tuple[float, float] = (0.9, 0.1)  # 训练集和验证集的分割比例
    random_seed: int = 2025  # 随机种子，用于数据分割
    batch_size: int = 1  # 批量大小
    num_workers: int = 2  # 数据加载的工作线程数


@dataclass
class DiTTrainingConfig:
    batch_size: int = 2  # 训练批量大小
    learning_rate: float = 3e-5  # 学习率
    num_epochs: int = 800  # 训练轮数
    warmup_epochs: int = 30  # 学习率预热轮数
    guidance_scale: float = 3.0  # 分类器引导尺度
