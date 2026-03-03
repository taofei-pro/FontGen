#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Infer LDM"

python infer_ldm.py \
  --sample_steps 50 \
  --output_dir data/outputs
