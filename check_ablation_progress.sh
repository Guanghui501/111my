#!/bin/bash

# ============================================================================
# 消融实验监控脚本
# 用于检查后台运行的消融实验状态
# ============================================================================

BASE_OUTPUT_DIR="./ablation_experiments"

echo "============================================================================"
echo "📊 消融实验状态检查"
echo "============================================================================"
echo ""

# 检查是否有后台进程
if pgrep -f "run_ablation_study.sh --background-mode" > /dev/null; then
    PID=$(pgrep -f "run_ablation_study.sh --background-mode")
    echo "✅ 实验正在运行中..."
    echo "   进程PID: $PID"
    echo ""
else
    echo "⚠️  未检测到运行中的实验进程"
    echo ""

    # 检查是否已完成
    if [ -f "$BASE_OUTPUT_DIR/COMPLETED" ]; then
        echo "✅ 实验已完成！"
        echo "   完成时间: $(cat $BASE_OUTPUT_DIR/COMPLETED)"
        echo ""
    fi
fi

# 检查各个实验目录
echo "实验进度:"
echo "----------"

experiments=(
    "exp1_text_concat_baseline:Exp-1 (Baseline)"
    "exp2_late_fusion:Exp-2 (Late Fusion)"
    "exp3_middle_fusion:Exp-3 (Middle Fusion)"
    "exp4_fine_grained:Exp-4 (Fine-Grained)"
    "exp5_full_model:Exp-5 (Full Model)"
)

for exp_info in "${experiments[@]}"; do
    IFS=':' read -r exp_dir exp_name <<< "$exp_info"
    exp_path="$BASE_OUTPUT_DIR/$exp_dir"

    if [ -d "$exp_path" ]; then
        if [ -f "$exp_path/history_val.json" ]; then
            # 获取训练轮数
            epochs=$(python3 -c "import json; f=open('$exp_path/history_val.json'); data=json.load(f); print(len(data.get('loss', [])))" 2>/dev/null || echo "?")
            echo "  ✅ $exp_name - 已完成 $epochs 轮"
        else
            echo "  🔄 $exp_name - 进行中..."
        fi
    else
        echo "  ⏸️  $exp_name - 未开始"
    fi
done

echo ""
echo "============================================================================"
echo "日志文件:"
echo "============================================================================"

# 显示最新的日志文件
if [ -f "ablation_nohup.log" ]; then
    echo "📝 Nohup日志: ablation_nohup.log"
    echo "   大小: $(du -h ablation_nohup.log | cut -f1)"
fi

latest_log=$(ls -t $BASE_OUTPUT_DIR/ablation_log_*.txt 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "📝 详细日志: $latest_log"
    echo "   大小: $(du -h $latest_log | cut -f1)"
    echo ""
    echo "最后10行:"
    echo "----------"
    tail -10 "$latest_log" | sed 's/^/   /'
fi

echo ""
echo "============================================================================"
echo "监控命令:"
echo "============================================================================"
echo "  ./check_ablation_progress.sh              # 再次运行此脚本"
echo "  tail -f ablation_nohup.log                # 实时查看nohup日志"
if [ -n "$latest_log" ]; then
    echo "  tail -f $latest_log  # 实时查看详细日志"
fi
if pgrep -f "run_ablation_study.sh --background-mode" > /dev/null; then
    echo "  kill $(pgrep -f "run_ablation_study.sh --background-mode")                                  # 终止实验"
fi
echo ""
