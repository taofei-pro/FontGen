#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQVAE"

python train_vqvae.py \
  --batch_size 8 \
  --lr 1e-3 \
  --num_epochs 100
