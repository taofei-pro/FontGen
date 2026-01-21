#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Build font from SVG"

SVG_DIR="${SVG_DIR:-svgs}"
OUTPUT_PATH="${OUTPUT_PATH:-fonts/output.ttf}"

python svg_to_font.py \
    --svg_dir "$SVG_DIR" \
    --output_path "$OUTPUT_PATH"
