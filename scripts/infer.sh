#!/usr/bin/env bash
set -euo pipefail

echo "[FontGen] Inference"

CMD=(python infer.py
    --condition_img_dir "data/target"
    --output_dir "data/outputs"
    --batch_size 4
    --num_workers 2
    --device "cuda"
    --vqgan2_ckpt "checkpoints/vqgan.pth"
    --dit_ckpt "checkpoints/dit.pth"
    --sampling_steps 50
    --guidance_scale 1.0
    --sampler "dpmpp_3m"
    --schedule "karras"
    --rho 7.0
    --cfg_rescale 0.4
    --x0_clip 1.5
    --use_component_mask
    --use_edge_map
    --enable_sr
    --sr_ckpt "checkpoints/sr_edsr.pth"
    --sr_model "edsr"
)

echo "Running: ${CMD[*]}"
${CMD[@]}
