#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train LDM"

python train_ldm.py \
  --batch_size 16 \
  --lr 5e-4 \
  --num_epochs 250 \
  --use_amp true \
  --split_ratios 0.9 0.1 \
  --random_seed 2025 \
  --img_save_interval 5 \
  --sample_steps 50
