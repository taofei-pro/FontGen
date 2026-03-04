#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Full training pipeline start"

# # 清理数据目录，避免旧数据干扰
# echo "[FontGen] Cleaning data directories..."
# rm -rf data/target/* data/reference/* data/outputs/* data/reference_infer/*

# # 1) 数据准备
# bash scripts/prepare_dataset.sh
# bash scripts/extract_charset.sh
# bash scripts/generate_reference.sh

# 2) 训练 VQVAE
bash scripts/train_vqvae.sh

# 3) 训练 LDM
bash scripts/train_ldm.sh

# 4) 训练 SR (暂时注释，因为 infer_ldm 不使用 SR 模型)
# bash scripts/train_sr.sh

# 5) 推理
bash scripts/infer_ldm.sh

# 6) 质量评估
bash scripts/compute_metrics.sh

echo "[FontGen] Full training pipeline done"
