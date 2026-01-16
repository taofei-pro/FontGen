#!/usr/bin/env bash
set -euo pipefail

echo "[NextGen] Inference"

CONDITION_IMG_DIR="${CONDITION_IMG_DIR:-data/target}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_nextgen}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"

VQGAN2_CKPT="${VQGAN2_CKPT:-checkpoints/vqgan2.pth}"
DIT_CKPT="${DIT_CKPT:-checkpoints/dit.pth}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
SAMPLER="${SAMPLER:-ddim}"
SCHEDULE="${SCHEDULE:-linear}"
RHO="${RHO:-7.0}"
CFG_RESCALE="${CFG_RESCALE:-0.0}"
X0_CLIP="${X0_CLIP:-}"

# Structure flags (set to 1 to enable)
USE_COMPONENT_MASK="${USE_COMPONENT_MASK:-1}"
USE_EDGE_MAP="${USE_EDGE_MAP:-1}"
USE_SKELETON="${USE_SKELETON:-0}"

# SR options
ENABLE_SR="${ENABLE_SR:-0}"
SR_CKPT="${SR_CKPT:-checkpoints/sr.pth}"
SR_MODEL="${SR_MODEL:-basic}"
SR_TILE="${SR_TILE:-0}"
SR_TILE_SIZE="${SR_TILE_SIZE:-256}"

CMD=(python infer_nextgen.py
    --condition_img_dir "$CONDITION_IMG_DIR"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --device "$DEVICE"
    --vqgan2_ckpt "$VQGAN2_CKPT"
    --dit_ckpt "$DIT_CKPT"
    --sampling_steps "$SAMPLING_STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --sampler "$SAMPLER"
    --schedule "$SCHEDULE"
    --rho "$RHO"
    --cfg_rescale "$CFG_RESCALE"
)

if [ -n "$X0_CLIP" ]; then
    CMD+=(--x0_clip "$X0_CLIP")
fi

if [ "$USE_COMPONENT_MASK" -eq 1 ]; then CMD+=(--use_component_mask); fi
if [ "$USE_EDGE_MAP" -eq 1 ]; then CMD+=(--use_edge_map); fi
if [ "$USE_SKELETON" -eq 1 ]; then CMD+=(--use_skeleton); fi

if [ "$ENABLE_SR" -eq 1 ]; then
    CMD+=(--enable_sr --sr_ckpt "$SR_CKPT" --sr_model "$SR_MODEL")
    if [ "$SR_TILE" -eq 1 ]; then
        CMD+=(--sr_tile --sr_tile_size "$SR_TILE_SIZE")
    fi
fi

echo "Running: ${CMD[*]}"
${CMD[@]}
