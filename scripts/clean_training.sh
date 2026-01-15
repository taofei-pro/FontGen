#!/bin/bash

# 训练产物清理脚本
# 在重新训练前执行，清理之前的训练产物

echo "🧹 开始清理训练产物..."
echo "📁 只清理训练相关的文件，保留重要数据"
echo ""

# 获取目标字体名称
TARGET_FONT_PATH="fonts/Z1.ttf"
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

echo "📁 目标字体: $TARGET_FONT_NAME"
echo ""

# 记录清理开始时间
CLEAN_START_TIME=$(date +%s)

# 清理模型权重文件
echo "🗑️  清理模型权重..."
if [ -d "checkpoints" ]; then
    # 检查是否有模型文件
    if ls checkpoints/*.pth 1> /dev/null 2>&1; then
        rm -f checkpoints/vqvae_*.pth
        rm -f checkpoints/ldm_*.pth
        echo "   ✅ 已清理 checkpoints/ 中的模型权重"
    else
        echo "   ℹ️  checkpoints/ 中没有模型权重文件"
    fi
else
    echo "   ℹ️  checkpoints/ 目录不存在"
fi

# 清理训练日志
echo "🗑️  清理训练日志..."
if [ -d "runs" ]; then
    if [ -d "runs/VQVAE" ] && [ "$(ls -A runs/VQVAE)" ]; then
        rm -rf runs/VQVAE/*
        echo "   ✅ 已清理 runs/VQVAE/ 中的训练日志"
    else
        echo "   ℹ️  runs/VQVAE/ 目录为空或不存在"
    fi
    
    if [ -d "runs/LDM" ] && [ "$(ls -A runs/LDM)" ]; then
        rm -rf runs/LDM/*
        echo "   ✅ 已清理 runs/LDM/ 中的训练日志"
    else
        echo "   ℹ️  runs/LDM/ 目录为空或不存在"
    fi
else
    echo "   ℹ️  runs/ 目录不存在"
fi

# 清理生成样本
echo "🗑️  清理生成样本..."
if [ -d "samples_${TARGET_FONT_NAME}" ]; then
    if [ "$(ls -A samples_${TARGET_FONT_NAME})" ]; then
        rm -rf samples_${TARGET_FONT_NAME}/*
        echo "   ✅ 已清理 samples_${TARGET_FONT_NAME}/ 中的生成样本"
    else
        echo "   ℹ️  samples_${TARGET_FONT_NAME}/ 目录为空"
    fi
else
    echo "   ℹ️  samples_${TARGET_FONT_NAME}/ 目录不存在"
fi

# 清理SVG输出（如果存在）
echo "🗑️  清理SVG输出..."
if [ -d "svgs_${TARGET_FONT_NAME}" ]; then
    if [ "$(ls -A svgs_${TARGET_FONT_NAME})" ]; then
        rm -rf svgs_${TARGET_FONT_NAME}/*
        echo "   ✅ 已清理 svgs_${TARGET_FONT_NAME}/ 中的SVG文件"
    else
        echo "   ℹ️  svgs_${TARGET_FONT_NAME}/ 目录为空"
    fi
else
    echo "   ℹ️  svgs_${TARGET_FONT_NAME}/ 目录不存在"
fi

# 清理字集划分（需要重新生成）
echo "🗑️  清理字集划分..."
if [ -d "charsets/splits/${TARGET_FONT_NAME}" ]; then
    if [ -f "charsets/splits/${TARGET_FONT_NAME}/train.txt" ] || [ -f "charsets/splits/${TARGET_FONT_NAME}/val.txt" ]; then
        rm -f charsets/splits/${TARGET_FONT_NAME}/train.txt
        rm -f charsets/splits/${TARGET_FONT_NAME}/val.txt
        echo "   ✅ 已清理字集划分文件"
    else
        echo "   ℹ️  字集划分文件不存在"
    fi
else
    echo "   ℹ️  字集划分目录不存在"
fi

# 清理Python缓存（只清理训练相关的缓存）
echo "🗑️  清理Python缓存..."
if [ -d "__pycache__" ]; then
    rm -rf __pycache__
    echo "   ✅ 已清理 __pycache__/"
fi

if [ -d "utils/__pycache__" ]; then
    rm -rf utils/__pycache__
    echo "   ✅ 已清理 utils/__pycache__/"
fi

if [ -d "models/__pycache__" ]; then
    rm -rf models/__pycache__/
    echo "   ✅ 已清理 models/__pycache__/"
fi

if [ -d "datasets/__pycache__" ]; then
    rm -rf datasets/__pycache__/
    echo "   ✅ 已清理 datasets/__pycache__/"
fi

if [ -d "configs/__pycache__" ]; then
    rm -rf configs/__pycache__/
    echo "   ✅ 已清理 configs/__pycache__/"
fi

# 清理训练日志文件
echo "🗑️  清理训练日志文件..."
if [ -d "logs" ]; then
    if [ "$(ls -A logs)" ]; then
        rm -f logs/*.log
        rm -f logs/*.txt
        echo "   ✅ 已清理 logs/ 中的日志文件"
    else
        echo "   ℹ️  logs/ 目录为空"
    fi
else
    echo "   ℹ️  logs/ 目录不存在"
fi

# 重要提醒：保留的内容
echo ""
echo "🔒 重要提醒：以下内容将被保留（不会删除）："
echo "   - 覆盖率分析结果 (charsets/jf7000_coverage/, charsets/unihan_coverage/)"
echo "   - 数据集图像 (data/ 目录)"
echo "   - 字体文件 (fonts/ 目录)"
echo "   - 配置文件 (configs/ 目录)"
echo "   - 源代码和脚本"
echo "   - 用户自定义数据"
echo ""

# 计算清理耗时
CLEAN_END_TIME=$(date +%s)
CLEAN_DURATION=$((CLEAN_END_TIME - CLEAN_START_TIME))

echo "🎉 清理完成！"
echo "⏰ 清理耗时: ${CLEAN_DURATION}秒"
echo ""
echo "📋 已清理的训练产物："
echo "   - 模型权重文件 (checkpoints/*.pth)"
echo "   - 训练日志 (runs/VQVAE/*, runs/LDM/*)"
echo "   - 生成样本 (samples_${TARGET_FONT_NAME}/*)"
echo "   - SVG输出 (svgs_${TARGET_FONT_NAME}/*)"
echo "   - 字集划分 (charsets/splits/${TARGET_FONT_NAME}/*)"
echo "   - Python缓存 (__pycache__/)"
echo "   - 训练日志文件 (logs/*)"
echo ""
echo "🚀 现在可以开始重新训练了！"
echo "   推荐执行顺序："
echo "   1. bash scripts/full_training_pipeline.sh  (完整流程)"
echo "   或者分步执行："
echo "   2. bash scripts/extract_charset.sh"
echo "   3. bash scripts/train_vqvae.sh"
echo "   4. bash scripts/train_ldm.sh"
echo ""
echo "⏰ 清理完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
