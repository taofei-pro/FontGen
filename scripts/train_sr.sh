#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train SR"

python train_sr.py \
  --model_name edsr \
  --max_steps 10000 \
  --save_path checkpoints/sr_edsr.pth
