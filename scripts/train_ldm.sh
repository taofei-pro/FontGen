#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train LDM"

python train_ldm.py \
  --batch_size 8 \
  --lr 1e-4 \
  --num_epochs 1000 \
  --use_amp true \
  --split_ratios 0.8 0.2 \
  --random_seed 2025 \
  --img_save_interval 5 \
  --sample_steps 300
