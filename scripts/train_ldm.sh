#!/bin/bash

TARGET_FONT_PATH="fonts/Z1.ttf"
TRAIN_SPLIT_RATIO=0.9           # 保持0.9/0.1比例
VAL_SPLIT_RATIO=0.1             # 保持0.9/0.1比例
RANDOM_SEED=2025
BATCH_SIZE=4                     # 降低批次大小，避免显存不足（OOM优化）
LEARNING_RATE=3e-4               # 降低学习率，提高稳定性（大数据集优化）
NUM_EPOCHS=450                   # 减少训练轮数（数据量增加12.7倍）
SAMPLE_STEPS=200                 # 进一步增加采样步数，提升生成质量
IMG_SAVE_INTERVAL=5              # 保持5
LPIPS_EVAL_INTERVAL=10           # 保持10
EVAL_BATCH_SIZE=4                # 保持4
DEVICE="cuda"

echo "🚀 开始基于VQ-GAN的LDM训练..."
echo "📊 数据集大小: 9,497个字符（训练集: 8,547 | 验证集: 950）"
echo "🎯 优化策略: 基于VQ-GAN的高质量潜在空间 + 增强扩散模型"
echo "💾 显存优化: 针对RTX 5090D优化"
echo "🔧 架构改进: VQ-GAN潜在空间 + 改进的注意力机制 + 残差连接"
echo "🚀 预期提升: 利用VQ-GAN的优质潜在表示提升生成质量"
echo ""

# 获取目标字体名称
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

# 清理LDM相关的训练产物
echo "🧹 清理LDM训练产物..."
if [ -f "checkpoints/ldm_${TARGET_FONT_NAME}.pth" ]; then
    rm -f checkpoints/ldm_${TARGET_FONT_NAME}.pth
    echo "   ✅ 已清理旧的LDM模型权重"
fi

if [ -d "runs/LDM" ] && [ "$(ls -A runs/LDM)" ]; then
    rm -rf runs/LDM/*
    echo "   ✅ 已清理LDM训练日志"
fi

if [ -d "samples_${TARGET_FONT_NAME}" ] && [ "$(ls -A samples_${TARGET_FONT_NAME})" ]; then
    rm -rf samples_${TARGET_FONT_NAME}/*
    echo "   ✅ 已清理LDM生成样本"
fi

if [ -d "svgs_${TARGET_FONT_NAME}" ] && [ "$(ls -A svgs_${TARGET_FONT_NAME})" ]; then
    rm -rf svgs_${TARGET_FONT_NAME}/*
    echo "   ✅ 已清理SVG输出文件"
fi

# 清理Python缓存
if [ -d "__pycache__" ]; then
    rm -rf __pycache__
fi
if [ -d "models/__pycache__" ]; then
    rm -rf models/__pycache__/
fi
if [ -d "datasets/__pycache__" ]; then
    rm -rf datasets/__pycache__/
fi
if [ -d "configs/__pycache__" ]; then
    rm -rf configs/__pycache__/
fi

echo "✅ LDM清理完成，开始训练..."
echo ""

PRETRAINED_VQVAE_PATH="checkpoints/vqgan_${TARGET_FONT_NAME}.pth"
MODEL_SAVE_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

echo "📁 目标字体: $TARGET_FONT_NAME"
echo "💾 VQ-GAN模型: $PRETRAINED_VQVAE_PATH"
echo "💾 LDM模型保存路径: $MODEL_SAVE_PATH"
echo "⚙️  训练参数（大数据集优化 + 显存优化）:"
echo "   - 批次大小: $BATCH_SIZE (显存优化，避免OOM)"
echo "   - 学习率: $LEARNING_RATE (降低学习率，提高稳定性)"
echo "   - 训练轮数: $NUM_EPOCHS (数据量增加12.7倍，每个epoch数据更多)"
echo "   - 采样步数: $SAMPLE_STEPS"
echo "   - 训练/验证比例: $TRAIN_SPLIT_RATIO/$VAL_SPLIT_RATIO"
echo "   - 预热轮数: 60 (增加预热期，让模型稳定启动)"
echo "   - 早停耐心: 120 (大数据集需要更多耐心)"
echo "   - 数据增强: 已禁用"
echo "🔧 LDM架构速度优化:"
echo "   - UNet通道: 96 (回退到稳定值)"
echo "   - 时间嵌入: 1792"
echo "   - 时间步数: 1400"
echo "   - Stable Diffusion通道: 320"
echo "   - 残差连接: 启用 ✓"
echo "   - 注意力机制: 启用 ✓"
echo ""

# 检查VQ-VAE模型是否存在
if [ ! -f "$PRETRAINED_VQVAE_PATH" ]; then
    echo "❌ 错误: VQ-VAE模型文件不存在: $PRETRAINED_VQVAE_PATH"
    echo "请先运行 VQ-VAE 训练脚本"
    exit 1
fi

echo "✅ VQ-VAE模型检查通过，开始LDM训练..."
echo ""

python train_ldm.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --pretrained_vqvae_path "$PRETRAINED_VQVAE_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_save_interval "$IMG_SAVE_INTERVAL" \
    --lpips_eval_interval "$LPIPS_EVAL_INTERVAL" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --use_stable_diffusion

echo ""
echo "✅ LDM训练完成！"
echo "📁 模型已保存到: $MODEL_SAVE_PATH"
echo "🖼️  生成样本保存在: $SAMPLE_ROOT"
