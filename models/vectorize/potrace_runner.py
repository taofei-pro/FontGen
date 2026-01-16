from __future__ import annotations

from pathlib import Path

from configs.converting_config import ConvertingConfig
from utils.image.image_converter import GlyphImageConverter


def raster_to_svg(
    image_path: str | Path,
    svg_path: str | Path,
    converting_config: ConvertingConfig | None = None,
) -> Path:
    """Convert a raster glyph image to SVG via potrace."""
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    converter = GlyphImageConverter(converting_config or ConvertingConfig())
    converter.convert_to_svg(image_path, svg_path)
    return svg_path
