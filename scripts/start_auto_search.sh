#!/bin/bash

# 自动化参数搜索启动脚本
# 用法: bash scripts/start_auto_search.sh [最大迭代次数]

echo "🚀 启动自动化参数搜索系统"
echo "================================"

# 设置默认参数
MAX_ITERATIONS=${1:-20}
TARGET_FONT="fonts/Z1.ttf"

echo "目标字体: $TARGET_FONT"
echo "最大迭代次数: $MAX_ITERATIONS"
echo ""

# 检查必要文件
if [ ! -f "scripts/auto_parameter_search.py" ]; then
    echo "❌ 错误: 自动化搜索脚本不存在"
    exit 1
fi

if [ ! -f "scripts/full_training_pipeline.sh" ]; then
    echo "❌ 错误: 训练流程脚本不存在"
    exit 1
fi

echo "✅ 所有必要文件检查通过"
echo ""

# 直接启动，无需确认
echo "⚠️  系统将自动执行多轮训练，每次约1-2小时"
echo "   总预计时间: $((MAX_ITERATIONS * 2)) 小时"
echo ""

echo ""
echo "🚀 启动自动化参数搜索..."
echo "📊 实时日志: auto_search.log"
echo "📈 搜索历史: search_history_Z1.json"
echo "📝 训练日志: TRAINING_LOG.md"
echo ""

# 启动搜索
python scripts/auto_parameter_search.py \
    --target_font_path "$TARGET_FONT" \
    --max_iterations "$MAX_ITERATIONS"

echo ""
echo "🎉 自动化参数搜索完成！"
echo "请查看以下文件了解结果："
echo "- auto_search.log: 详细执行日志"
echo "- search_history_Z1.json: 搜索历史记录"
echo "- TRAINING_LOG.md: 训练批次记录"
