#!/usr/bin/env bash
set -euo pipefail

echo "[NextGen] Full pipeline start"

bash scripts/train_vqgan2.sh
bash scripts/train_dit.sh
bash scripts/train_sr.sh

python infer_nextgen.py

echo "[NextGen] Full pipeline done"
