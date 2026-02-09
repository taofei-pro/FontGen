#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Full training pipeline start"

# 1) 数据准备
bash scripts/prepare_dataset.sh
bash scripts/extract_charset.sh

# 2) 训练 VQGAN
bash scripts/train_vqgan.sh

# 3) 训练 DiT
bash scripts/train_dit.sh

# 4) 训练 SR
bash scripts/train_sr.sh

# 5) 推理（已启用 SR）
bash scripts/infer.sh

# 6) 质量评估（可选）
bash scripts/compute_metrics.sh

echo "[FontGen] Full training pipeline done"
