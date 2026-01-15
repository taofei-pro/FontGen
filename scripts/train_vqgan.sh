#!/bin/bash

# VQ-GAN Training Script
# 在现有VQ-VAE基础上启用VQ-GAN模式

set -e

# 字体路径配置
TARGET_FONT_PATH="fonts/Z1.ttf"

# 检查字体文件
if [ ! -f "$TARGET_FONT_PATH" ]; then
    echo "❌ 错误: 字体文件不存在: $TARGET_FONT_PATH"
    echo "请确保字体文件存在，或修改TARGET_FONT_PATH变量"
    exit 1
fi

echo "🚀 开始VQ-GAN训练..."
echo "📊 数据集大小: 9,497个字符（训练集: 8,547 | 验证集: 950）"
echo "🎯 优化策略: 大数据集优化 + 对抗训练 + 感知损失"
echo "💾 显存优化: 针对RTX 5090D优化"
echo "🔧 架构改进: VQ-VAE + PatchGAN判别器 + 感知损失"
echo ""

# 获取目标字体名称
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf|ttc)$//')

# 清理VQ-GAN相关的训练产物
echo "🧹 清理VQ-GAN训练产物..."
if [ -f "checkpoints/vqgan_${TARGET_FONT_NAME}.pth" ]; then
    rm -f checkpoints/vqgan_${TARGET_FONT_NAME}.pth
    echo "   ✅ 已清理旧的VQ-GAN模型权重"
fi

if [ -d "runs/VQGAN" ] && [ "$(ls -A runs/VQGAN)" ]; then
    rm -rf runs/VQGAN/*
    echo "   ✅ 已清理VQ-GAN训练日志"
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

echo "✅ VQ-GAN清理完成，开始训练..."
echo ""

MODEL_SAVE_PATH="checkpoints/vqgan_${TARGET_FONT_NAME}.pth"

echo "📁 目标字体: $TARGET_FONT_NAME"
echo "💾 模型保存路径: $MODEL_SAVE_PATH"
echo "🔧 VQ-GAN配置:"
echo "   - 基础架构: VQ-VAE (速度优化版)"
echo "   - 编码器通道: 96 (回退到稳定值)"
echo "   - 潜在维度: 4 (回退到稳定值)"
echo "   - 码本大小: 192"
echo "   - 模型参数: 17.36M (减少27.6%)"
echo "   - 残差连接: 启用 ✓"
echo "   - 注意力机制: 启用 ✓"
echo "   - 跳跃连接: 暂时禁用（显存优化）"
echo "   - 判别器: PatchGAN"
echo "   - 感知损失: 启用 (权重0.4)"
echo "   - 对抗训练: 启用 (权重0.1)"
echo "   - 批次大小: 4 (显存优化，避免OOM)"
echo "   - 训练轮数: 300 (数据量增加12.7倍，每个epoch数据更多)"
echo "   - 学习率: 6e-4 (降低学习率，提高稳定性)"
echo "   - 预热轮数: 40 (增加预热期，让模型稳定启动)"
echo "   - 早停耐心: 120 (大数据集需要更多耐心)"
echo "   - 数据增强: 已禁用"
echo ""

# 创建必要的目录
mkdir -p checkpoints
mkdir -p runs/VQGAN
mkdir -p samples_${TARGET_FONT_NAME}

# 训练VQ-GAN
echo "🎯 开始VQ-GAN训练..."
python train_vqgan.py \
    --font_path "$TARGET_FONT_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --log_dir "runs/VQGAN" \
    --sample_dir "samples_${TARGET_FONT_NAME}" \
    --num_epochs 300 \
    --batch_size 4 \
    --learning_rate 6e-4 \
    --use_vqgan \
    --discriminator_lr 2e-4 \
    --perceptual_weight 0.4 \
    --adversarial_weight 0.1 \
    --save_every_n_epochs 50 \
    --log_every_n_steps 100 \
    --val_every_n_epochs 10

# 检查训练结果
if [ -f "$MODEL_SAVE_PATH" ]; then
    echo ""
    echo "✅ VQ-GAN训练完成！"
    echo "📁 模型保存位置: $MODEL_SAVE_PATH"
    echo "📊 训练日志: runs/VQGAN"
    echo "🎨 生成样本: samples_${TARGET_FONT_NAME}"
    echo ""
    echo "🚀 下一步: 使用VQ-GAN训练LDM模型"
    echo "   运行: bash scripts/train_ldm.sh"
    echo ""
    echo "📈 预期性能提升:"
    echo "   - PSNR: +35-60% (11.23 → 15-18)"
    echo "   - SSIM: +8-14% (0.83 → 0.90-0.95)"
    echo "   - LPIPS: -43-64% (0.14 → 0.05-0.08)"
    echo "   - FID: -56-73% (56.47 → 15-25)"
else
    echo ""
    echo "❌ VQ-GAN训练失败！"
    echo "请检查错误日志并重试"
    exit 1
fi
