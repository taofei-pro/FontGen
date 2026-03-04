from dataclasses import dataclass


@dataclass
class VQGAN2DatasetConfig:
    target_img_dir: str = "data/target"  # 目标字体图像目录
    reference_img_dir: str = "data/reference"  # 参考字体图像目录
    split_ratios: tuple[float, float] = (0.9, 0.1)  # 训练集和验证集的分割比例
    random_seed: int = 2025  # 随机种子，用于数据分割
    batch_size: int = 1  # 批量大小
    num_workers: int = 2  # 数据加载的工作线程数


@dataclass
class VQGAN2ModelConfig:
    input_img_channels: int = 1  # 输入图像的通道数，1表示灰度图
    base_channels: int = 64  # 模型基础通道数
    latent_dim: int = 32  # 潜在空间的维度
    token_dim: int = 256  # Token的维度
    codebook_size: int = 512  # 码本大小，决定了离散表示的词汇量
    commitment_cost: float = 0.1  # 承诺损失的权重
    multiscale: bool = False  # 是否使用多尺度量化
    coarse_downsample: int = 2  # 粗尺度下采样因子
    coarse_weight: float = 0.5  # 粗尺度损失的权重
    tanh_output: bool = True  # 是否使用tanh激活函数约束输出范围
    use_vqgan: bool = True  # 是否使用VQGAN
    vq_decay: float = 0.99  # 向量量化的衰减系数
    vq_epsilon: float = 1e-5  # 向量量化的epsilon值


@dataclass
class VQGAN2TrainingConfig:
    batch_size: int = 2  # 训练批量大小
    learning_rate: float = 2e-5  # 学习率
    num_epochs: int = 300  # 训练轮数
    warmup_epochs: int = 30  # 学习率预热轮数
    weight_decay: float = 1e-4  # 权重衰减系数
    discriminator_lr: float = 1e-4  # 判别器的学习率
    perceptual_weight: float = 0.3  # 感知损失的权重
    foreground_weight: float = 2.0  # 前景加权损失的权重
    adversarial_weight: float = 0.05  # 对抗损失的权重
    discriminator_start_steps: int = 2000  # 判别器开始训练的步数
    model_save_path: str = "checkpoints/vqgan.pth"  # 模型保存路径