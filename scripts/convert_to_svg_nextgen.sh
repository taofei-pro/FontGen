#!/usr/bin/env bash
set -euo pipefail

echo "[NextGen] Convert outputs to SVG"

INPUT_DIR="${INPUT_DIR:-outputs_nextgen}"
OUTPUT_DIR="${OUTPUT_DIR:-svgs_nextgen}"

BLACKLEVEL="${BLACKLEVEL:-0.5}"
TURDSIZE="${TURDSIZE:-2}"
ALPHAMAX="${ALPHAMAX:-1.0}"
OPTTOLERANCE="${OPTTOLERANCE:-0.2}"

python convert_to_svg.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --blacklevel "$BLACKLEVEL" \
    --turdsize "$TURDSIZE" \
    --alphamax "$ALPHAMAX" \
    --opttolerance "$OPTTOLERANCE"
