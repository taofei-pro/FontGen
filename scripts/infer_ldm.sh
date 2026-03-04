#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Infer LDM"

python infer_ldm.py \
  --sample_steps 50 \
  --output_dir data/outputs \
  --charset_path charsets/target_charset.txt \
  --ref_dir data/reference_infer
