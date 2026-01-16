#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
TRAIN_SPLIT_RATIO=0.9           # 从0.8增加到0.9，增加训练集
VAL_SPLIT_RATIO=0.1             # 从0.2降到0.1，减少验证集
RANDOM_SEED=2025
DEVICE="cuda"


python extract_charset.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --device "$DEVICE"
