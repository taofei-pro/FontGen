from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def svg_to_font(svg_dir: str | Path, output_path: str | Path) -> Path:
    """Build a font file from an SVG directory using FontForge (if available)."""
    svg_dir = Path(svg_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not svg_dir.is_dir():
        raise ValueError(f"SVG directory not found: {svg_dir}")

    script = f"""
import fontforge
import os
from pathlib import Path

svg_dir = Path(r"{svg_dir}")
out_path = Path(r"{output_path}")

font = fontforge.font()
for svg_path in sorted(svg_dir.glob("*.svg")):
    name = svg_path.stem
    try:
        codepoint = int(name, 16)
    except ValueError:
        continue
    glyph = font.createChar(codepoint)
    glyph.importOutlines(str(svg_path))
    glyph.width = 1000

font.generate(str(out_path))
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
        fp.write(script)
        script_path = fp.name

    result = subprocess.run(
        ["fontforge", "-lang=py", "-script", script_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "FontForge failed. Make sure the `fontforge` command is available.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return output_path
