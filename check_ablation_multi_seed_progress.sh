#!/bin/bash

# ============================================================================
# 多种子消融实验监控脚本（串行执行版本）
# 检查4个实验 × 3个种子 = 12个训练任务的状态
# ============================================================================

BASE_OUTPUT_DIR="./ablation_multi_seed"

echo "============================================================================"
echo "📊 消融实验状态检查（串行执行版本）"
echo "============================================================================"
echo ""
echo "时间: $(date)"
echo ""

# ============================================================================
# 1. 各实验详细进度
# ============================================================================
echo "============================================================================"
echo "1️⃣  实验详细进度"
echo "============================================================================"
echo ""

# 定义实验配置
declare -A exp_names=(
    [1]="Exp-1: Baseline"
    [2]="Exp-2: +Late Fusion"
    [3]="Exp-3: +Middle Fusion"
    [4]="Exp-4: +Fine-Grained"
)

seeds=(42 123 7)

for exp_num in {1..4}; do
    echo "----------------------------------------"
    echo "${exp_names[$exp_num]}"
    echo "----------------------------------------"

    for seed in "${seeds[@]}"; do
        exp_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"

        if [ -d "$exp_dir" ]; then
            # 检查训练历史文件
            if [ -f "$exp_dir/history_val.json" ]; then
                # 使用Python获取当前轮数和最佳性能
                epoch_info=$(python3 -c "
import json
import sys
try:
    with open('$exp_dir/history_val.json', 'r') as f:
        data = json.load(f)
    epochs = len(data.get('loss', []))

    # 检测任务类型
    if 'mae' in data:
        metric = 'mae'
        best_val = min(data[metric])
        last_val = data[metric][-1]
    elif 'accuracy' in data:
        metric = 'accuracy'
        best_val = max(data[metric])
        last_val = data[metric][-1]
    else:
        metric = 'unknown'
        best_val = 0
        last_val = 0

    print(f'{epochs}|{metric}|{best_val:.4f}|{last_val:.4f}')
except:
    print('0|unknown|0|0')
" 2>/dev/null)

                IFS='|' read -r epochs metric best_val last_val <<< "$epoch_info"

                if [ "$epochs" != "0" ]; then
                    echo "  ✅ Seed $seed: 已完成 $epochs 轮"
                    echo "     最佳 $metric: $best_val | 最后 $metric: $last_val"
                else
                    echo "  🔄 Seed $seed: 进行中..."
                fi
            else
                # 检查training.log是否有内容
                if [ -f "$exp_dir/training.log" ]; then
                    log_size=$(du -h "$exp_dir/training.log" | cut -f1)
                    echo "  🔄 Seed $seed: 进行中... (日志大小: $log_size)"
                else
                    echo "  ⏳ Seed $seed: 准备启动..."
                fi
            fi
        else
            echo "  ⏸️  Seed $seed: 未开始"
        fi
    done

    echo ""
done

# ============================================================================
# 2. 最新日志摘要
# ============================================================================
echo "============================================================================"
echo "2️⃣  最新日志摘要（各实验最后5行）"
echo "============================================================================"
echo ""

for exp_num in {1..4}; do
    echo "----------------------------------------"
    echo "${exp_names[$exp_num]}"
    echo "----------------------------------------"

    for seed in "${seeds[@]}"; do
        log_file="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}/training.log"

        if [ -f "$log_file" ] && [ -s "$log_file" ]; then
            echo ""
            echo "  📝 Seed $seed (最后5行):"
            tail -5 "$log_file" | sed 's/^/     /'
        fi
    done

    echo ""
done

# ============================================================================
# 3. 结果汇总表
# ============================================================================
echo "============================================================================"
echo "3️⃣  结果汇总表"
echo "============================================================================"
echo ""

# 表头
printf "%-25s | %-12s | %-12s | %-12s\n" "实验配置" "Seed 42" "Seed 123" "Seed 7"
echo "--------------------------------------------------------------------------------"

for exp_num in {1..4}; do
    exp_name="${exp_names[$exp_num]}"

    # 缩短实验名称以适应表格
    case $exp_num in
        1) short_name="Baseline" ;;
        2) short_name="+Late Fusion" ;;
        3) short_name="+Middle Fusion" ;;
        4) short_name="+Fine-Grained" ;;
    esac

    results=()
    for seed in "${seeds[@]}"; do
        exp_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"

        if [ -f "$exp_dir/history_val.json" ]; then
            result=$(python3 -c "
import json
try:
    with open('$exp_dir/history_val.json', 'r') as f:
        data = json.load(f)

    if 'mae' in data:
        metric = 'MAE'
        best_val = min(data['mae'])
    elif 'accuracy' in data:
        metric = 'Acc'
        best_val = max(data['accuracy'])
    else:
        metric = '?'
        best_val = 0

    print(f'{metric}:{best_val:.4f}')
except:
    print('N/A')
" 2>/dev/null)
            results+=("$result")
        else
            results+=("Running...")
        fi
    done

    printf "%-25s | %-12s | %-12s | %-12s\n" \
        "$short_name" \
        "${results[0]}" \
        "${results[1]}" \
        "${results[2]}"
done

echo ""

# ============================================================================
# 4. 磁盘使用情况
# ============================================================================
echo "============================================================================"
echo "4️⃣  磁盘使用情况"
echo "============================================================================"
echo ""

if [ -d "$BASE_OUTPUT_DIR" ]; then
    total_size=$(du -sh "$BASE_OUTPUT_DIR" | cut -f1)
    echo "  总大小: $total_size"
    echo ""
    echo "  各实验大小:"

    for exp_num in {1..4}; do
        exp_total=0
        for seed in "${seeds[@]}"; do
            exp_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"
            if [ -d "$exp_dir" ]; then
                size=$(du -sm "$exp_dir" | cut -f1)
                exp_total=$((exp_total + size))
            fi
        done

        if [ $exp_total -gt 0 ]; then
            echo "    ${exp_names[$exp_num]}: ${exp_total}MB"
        fi
    done
    echo ""
fi

# ============================================================================
# 5. 快捷监控命令
# ============================================================================
echo "============================================================================"
echo "📝 快捷监控命令"
echo "============================================================================"
echo ""
echo "  查看特定实验日志 (例如 Exp1, Seed42):"
echo "    tail -f $BASE_OUTPUT_DIR/exp1_seed42/training.log"
echo ""
echo "  查看主启动日志:"
echo "    tail -f $BASE_OUTPUT_DIR/launch_log_*.txt"
echo ""
echo "  查看GPU使用:"
echo "    nvidia-smi"
echo ""
echo "  实时监控此脚本:"
echo "    watch -n 60 ./check_ablation_multi_seed_progress.sh"
echo ""
echo "  查看当前正在训练的实验 (查找python进程):"
echo "    ps aux | grep train_with_cross_modal_attention.py"
echo ""
echo "============================================================================"
echo ""
