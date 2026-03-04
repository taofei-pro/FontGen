from dataclasses import dataclass


@dataclass
class ConvertingConfig:
    """
    Configuration class for converting settings.
    """

    blacklevel: float = 0.5  # 黑色阈值，用于二值化图像
    turdsize: int = 2  # 小斑点的大小阈值，小于此值的斑点会被忽略
    alphamax: float = 1  # 拐角的最大角度
    opttolerance: float = 0.2  # 优化容差，控制曲线拟合的精度
