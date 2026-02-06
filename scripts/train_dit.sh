#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Train DiT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_dit.py \
  --vqgan_ckpt checkpoints/vqgan.pth \
  --max_steps 10000 \
  --batch_size 1 \
  --num_workers 0
