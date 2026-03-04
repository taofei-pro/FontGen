#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQVAE"

python train_vqvae.py \
  --batch_size 8 \
  --lr 1e-3 \
  --num_epochs 100 \
  --use_amp true \
  --split_ratios 0.9 0.1 \
  --random_seed 2025 \
  --img_save_interval 5
