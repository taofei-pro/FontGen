#!/bin/bash
set -eu

echo "[FontGen] Train DiT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_dit.py \
  --vqgan_ckpt checkpoints/vqgan.pth \
  --use_component_mask \
  --use_edge_map \
  --max_steps 100000

