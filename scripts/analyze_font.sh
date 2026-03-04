#!/bin/bash

TARGET_FONT_PATH="fonts/M8.ttf"
REFERENCE_FONTS_DIR="fonts/reference/"


python analyze_font.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --analyze_target_font \
    --analyze_reference_fonts
