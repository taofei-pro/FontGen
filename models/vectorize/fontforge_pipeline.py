from __future__ import annotations

from pathlib import Path


def svg_to_font(svg_dir: str | Path, output_path: str | Path) -> Path:
    """Placeholder: build font file from SVG directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Actual FontForge pipeline will be added in implementation phase.
    return output_path
