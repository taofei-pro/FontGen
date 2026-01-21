#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train VQGAN-2"

python train_vqgan2.py \
  --max_steps 5000 \
  --perceptual_weight 0.4 \
  --adversarial_weight 0.1 \
  --discriminator_start_steps 500
