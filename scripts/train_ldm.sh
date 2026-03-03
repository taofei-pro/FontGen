#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train LDM"

python train_ldm.py \
  --batch_size 16 \
  --lr 5e-4 \
  --num_epochs 250
