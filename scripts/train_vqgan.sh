#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQGAN-2"

python train_vqgan.py \
  --max_steps 40000 \
  --perceptual_weight 0.4 \
  --adversarial_weight 0.05 \
  --discriminator_start_steps 1000
