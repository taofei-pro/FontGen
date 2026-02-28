#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQGAN-2"

python train_vqgan.py \
  --max_steps 45000 \
  --perceptual_weight 0.5 \
  --adversarial_weight 0.15 \
  --discriminator_start_steps 1000
