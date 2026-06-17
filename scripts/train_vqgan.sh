#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQGAN-2 (Optimized)"

python train_vqgan.py \
  --max_steps 80000 \
  --perceptual_weight 0.5 \
  --adversarial_weight 0.1 \
  --discriminator_start_steps 1000
