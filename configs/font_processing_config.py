from dataclasses import dataclass
from pathlib import Path


@dataclass
class FontProcessingConfig:
    """
    Configuration class for font analysis and dataset preparation settings.
    """

    charset_root: str = "charsets"  # 字符集根目录

    data_root: str = "data"  # 数据根目录
    img_size: tuple[int, int] = (512, 512)  # 图像大小
    sample_ratio: float = 1.0  # 采样比率，1.0表示使用全部数据
    num_workers: int = 4  # 数据处理的工作线程数

    @property
    def jf7000_charset_dir(self) -> Path:
        return Path(self.charset_root) / "jf7000"  # JF7000字符集目录

    @property
    def unihan_charset_dir(self) -> Path:
        return Path(self.charset_root) / "unihan"  # Unihan字符集目录

    @property
    def jf7000_charset_path(self) -> Path:
        return self.jf7000_charset_dir / "jf7000_all.txt"  # JF7000字符集文件路径

    @property
    def unihan_charset_path(self) -> Path:
        return self.unihan_charset_dir / "unihan_all.txt"  # Unihan字符集文件路径

    @property
    def jf7000_coverage_charset_dir(self) -> Path:
        return Path(self.charset_root) / "jf7000_coverage"  # JF7000覆盖率字符集目录

    @property
    def unihan_coverage_charset_dir(self) -> Path:
        return Path(self.charset_root) / "unihan_coverage"  # Unihan覆盖率字符集目录
