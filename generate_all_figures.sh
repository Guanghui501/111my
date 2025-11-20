#!/bin/bash

# ================================================================
# 一键生成所有论文图表
# 用法: ./generate_all_figures.sh <output_dir> [save_dir]
# ================================================================

if [ $# -lt 1 ]; then
    echo "用法: $0 <output_dir> [save_dir]"
    echo ""
    echo "参数说明:"
    echo "  output_dir  - 训练输出目录（包含history_*.json和predictions_*.csv）"
    echo "  save_dir    - 图片保存目录（可选，默认为output_dir/figures）"
    echo ""
    echo "示例:"
    echo "  $0 ./output_class_syn"
    echo "  $0 ./output_class_syn ./paper_figures"
    exit 1
fi

OUTPUT_DIR=$1
SAVE_DIR=${2:-"$OUTPUT_DIR/figures"}

echo "=========================================="
echo "📊 开始生成所有论文图表"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "保存目录: $SAVE_DIR"
echo ""

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 检查输出目录是否存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ 错误: 输出目录不存在: $OUTPUT_DIR"
    exit 1
fi

# ========== 1. 生成训练曲线 ==========
echo "📈 [1/2] 生成训练曲线..."
python plot_training_curves.py \
    --output_dir "$OUTPUT_DIR" \
    --save_dir "$SAVE_DIR" \
    --no_show

if [ $? -eq 0 ]; then
    echo "✅ 训练曲线生成成功"
else
    echo "⚠️  训练曲线生成失败"
fi
echo ""

# ========== 2. 生成预测结果图 ==========
echo "📈 [2/2] 生成预测结果图..."
python plot_predictions.py \
    --output_dir "$OUTPUT_DIR" \
    --save_dir "$SAVE_DIR" \
    --no_show

if [ $? -eq 0 ]; then
    echo "✅ 预测结果图生成成功"
else
    echo "⚠️  预测结果图生成失败"
fi
echo ""

# ========== 总结 ==========
echo "=========================================="
echo "✅ 图表生成完成！"
echo "=========================================="
echo "所有图片已保存到: $SAVE_DIR"
echo ""
echo "生成的文件："
ls -lh "$SAVE_DIR"/*.png 2>/dev/null | awk '{print "  - " $9}' | sed 's|.*/||'
ls -lh "$SAVE_DIR"/*.pdf 2>/dev/null | awk '{print "  - " $9}' | sed 's|.*/||'
echo ""
echo "提示: PDF格式适合论文投稿，PNG格式适合演示和预览"
echo "=========================================="
