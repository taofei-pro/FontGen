from __future__ import annotations

from pathlib import Path


def raster_to_svg(image_path: str | Path, svg_path: str | Path) -> Path:
    """Placeholder: convert raster image to SVG via potrace."""
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    # Actual potrace invocation will be added in implementation phase.
    return svg_path
