#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Test Infer LDM"

# 直接使用现有的 covered.txt 文件作为测试字符集
TEST_CHARSET_PATH="charsets/unihan_coverage/young/covered.txt"

# 测试推理（直接使用 data/reference 目录作为参考图像目录）
echo "[FontGen] Running test inference..."
python infer_ldm.py \
  --sample_steps 100 \
  --output_dir data/outputs_test \
  --charset_path "$TEST_CHARSET_PATH" \
  --ref_dir data/reference

# 测试评估
echo "[FontGen] Running test evaluation..."
python compute_metrics.py \
  --gen_dir data/outputs_test \
  --gt_dir data/target \
  --resize_gen_to_gt

echo "[FontGen] Test inference completed"
echo "[FontGen] Check results in data/outputs_test directory"
