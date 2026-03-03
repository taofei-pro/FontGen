#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Generate reference images"

# python extract_charset_from_font.py # 生成target_charset.txt

python generate_reference_images.py \
  --charset_path charsets/target_charset.txt \
  --reference_fonts_dir fonts/jigmo \
  --output_dir data/reference_infer
