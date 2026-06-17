#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQVAE"

python train_vqvae.py \
  --batch_size 8 \
  --lr 5e-4 \
  --num_epochs 200 \
  --use_amp true \
  --split_ratios 0.8 0.2 \
  --random_seed 2025 \
  --img_save_interval 5
