#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Full pipeline start"

bash scripts/train_vqgan.sh
bash scripts/train_dit.sh
bash scripts/train_sr.sh

bash scripts/infer.sh
bash scripts/convert_to_svg.sh
bash scripts/svg_to_font.sh

echo "[FontGen] Full pipeline done"
