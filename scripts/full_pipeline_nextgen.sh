#!/usr/bin/env bash
set -euo pipefail

echo "[NextGen] Full pipeline start"

bash scripts/train_vqgan2.sh
bash scripts/train_dit.sh
bash scripts/train_sr.sh

bash scripts/infer_nextgen.sh
bash scripts/convert_to_svg_nextgen.sh
bash scripts/svg_to_font.sh

echo "[NextGen] Full pipeline done"
