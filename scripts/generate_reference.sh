#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Generate reference images"

# 检查 target_charset.txt 是否存在，不存在则生成
if [ ! -f "charsets/target_charset.txt" ]; then
    echo "[FontGen] target_charset.txt not found, generating..."
    python extract_charset_from_font.py --font_path fonts/case/case.ttf --output_path charsets/target_charset.txt
fi

python generate_reference_images.py \
  --charset_path charsets/target_charset.txt \
  --reference_fonts_dir fonts/jigmo \
  --output_dir data/reference_infer
