#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Compute metrics"

python compute_metrics.py \
  --gen_dir "data/outputs" \
  --gt_dir "data/target" \
  --batch_size 8 \
  --resize_gen_to_gt
