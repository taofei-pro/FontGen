#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Inference"

# 环境已在终端中激活，无需再次激活

CMD=(python infer.py
    --condition_img_dir "data/target"
    --output_dir "data/outputs"
    --batch_size 1
    --num_workers 1
    --device "cuda"
    --vqgan_ckpt "checkpoints/vqgan.pth"
    --dit_ckpt "checkpoints/dit.pth"
    --sampling_steps 300 \
    --guidance_scale 9.0 \
    --sampler "dpmpp_3m" \
    --schedule "karras" \
    --rho 7.0 \
    --cfg_rescale 0.4 \
    --x0_clip 2.0 \
    --use_component_mask \
    --use_edge_map \
    --enable_sr \
    --sr_ckpt "checkpoints/sr_edsr.pth" \
    --sr_model "edsr" \
    --num_chars 20
)

echo "Running: ${CMD[*]}"
${CMD[@]}
