#!/bin/bash

# ============================================================================
# 全模块训练进度监控脚本
# 检查3个种子的Full Model训练状态
# ============================================================================

BASE_OUTPUT_DIR="./full_model_multi_seed"
PID_FILE="$BASE_OUTPUT_DIR/running_pids.txt"

echo "============================================================================"
echo "📊 Full Model训练状态检查"
echo "============================================================================"
echo ""
echo "时间: $(date)"
echo ""

# ============================================================================
# 1. 检查后台进程状态
# ============================================================================
echo "============================================================================"
echo "1️⃣  后台进程状态"
echo "============================================================================"
echo ""

if [ -f "$PID_FILE" ]; then
    mapfile -t PIDS < "$PID_FILE"

    running_count=0
    finished_count=0

    for pid in "${PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            running_count=$((running_count + 1))
        else
            finished_count=$((finished_count + 1))
        fi
    done

    total_count=${#PIDS[@]}
    echo "  总任务数: $total_count"
    echo "  运行中: $running_count"
    echo "  已完成: $finished_count"
    echo ""

    if [ $running_count -gt 0 ]; then
        echo "  运行中的进程PID:"
        for pid in "${PIDS[@]}"; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "    - PID $pid (运行时间: $(ps -p $pid -o etime= | xargs))"
            fi
        done
        echo ""
    fi
else
    echo "  ⚠️  未找到PID文件: $PID_FILE"
    echo "  可能训练尚未启动或PID文件已被删除"
    echo ""
fi

# ============================================================================
# 2. 训练详细进度
# ============================================================================
echo "============================================================================"
echo "2️⃣  训练详细进度"
echo "============================================================================"
echo ""

seeds=(42 123 7)

for seed in "${seeds[@]}"; do
    model_dir="$BASE_OUTPUT_DIR/full_model_seed${seed}"

    echo "----------------------------------------"
    echo "Full Model - Seed $seed"
    echo "----------------------------------------"

    if [ -d "$model_dir" ]; then
        # 检查训练历史文件
        if [ -f "$model_dir/history_val.json" ]; then
            # 使用Python获取当前轮数和最佳性能
            epoch_info=$(python3 -c "
import json
import sys
try:
    with open('$model_dir/history_val.json', 'r') as f:
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
                echo "  状态: ✅ 已完成 $epochs 轮"
                echo "  最佳 $metric: $best_val"
                echo "  最后 $metric: $last_val"
            else
                echo "  状态: 🔄 进行中..."
            fi
        else
            # 检查nohup.log是否有内容
            if [ -f "$model_dir/nohup.log" ]; then
                log_size=$(du -h "$model_dir/nohup.log" | cut -f1)
                echo "  状态: 🔄 进行中... (日志大小: $log_size)"
            else
                echo "  状态: ⏳ 准备启动..."
            fi
        fi

        # 检查模型文件
        if [ -f "$model_dir/best_model.pt" ]; then
            model_size=$(du -h "$model_dir/best_model.pt" | cut -f1)
            echo "  最佳模型: $model_size"
        fi

        echo "  输出目录: $model_dir"
    else
        echo "  状态: ⏸️  未开始"
    fi

    echo ""
done

# ============================================================================
# 3. 最新日志摘要
# ============================================================================
echo "============================================================================"
echo "3️⃣  最新日志摘要（各训练最后10行）"
echo "============================================================================"
echo ""

for seed in "${seeds[@]}"; do
    log_file="$BASE_OUTPUT_DIR/full_model_seed${seed}/nohup.log"

    if [ -f "$log_file" ] && [ -s "$log_file" ]; then
        echo "----------------------------------------"
        echo "Full Model - Seed $seed"
        echo "----------------------------------------"
        tail -10 "$log_file" | sed 's/^/  /'
        echo ""
    fi
done

# ============================================================================
# 4. 结果汇总表
# ============================================================================
echo "============================================================================"
echo "4️⃣  结果汇总表"
echo "============================================================================"
echo ""

printf "%-15s | %-12s | %-12s | %-12s\n" "Seed" "Epochs" "Best Metric" "Status"
echo "-----------------------------------------------------------"

for seed in "${seeds[@]}"; do
    model_dir="$BASE_OUTPUT_DIR/full_model_seed${seed}"

    if [ -f "$model_dir/history_val.json" ]; then
        result=$(python3 -c "
import json
try:
    with open('$model_dir/history_val.json', 'r') as f:
        data = json.load(f)

    epochs = len(data.get('loss', []))

    if 'mae' in data:
        metric = 'MAE'
        best_val = min(data['mae'])
    elif 'accuracy' in data:
        metric = 'Acc'
        best_val = max(data['accuracy'])
    else:
        metric = '?'
        best_val = 0

    # 检查是否完成
    if epochs >= 100:
        status = 'Completed'
    else:
        status = 'Running'

    print(f'{epochs}|{metric}:{best_val:.4f}|{status}')
except:
    print('0|N/A|Not Started')
" 2>/dev/null)

        IFS='|' read -r epochs metric status <<< "$result"
        printf "%-15s | %-12s | %-12s | %-12s\n" "$seed" "$epochs" "$metric" "$status"
    else
        printf "%-15s | %-12s | %-12s | %-12s\n" "$seed" "0" "N/A" "Not Started"
    fi
done

echo ""

# ============================================================================
# 5. 磁盘使用情况
# ============================================================================
echo "============================================================================"
echo "5️⃣  磁盘使用情况"
echo "============================================================================"
echo ""

if [ -d "$BASE_OUTPUT_DIR" ]; then
    total_size=$(du -sh "$BASE_OUTPUT_DIR" | cut -f1)
    echo "  总大小: $total_size"
    echo ""
    echo "  各训练大小:"

    for seed in "${seeds[@]}"; do
        model_dir="$BASE_OUTPUT_DIR/full_model_seed${seed}"
        if [ -d "$model_dir" ]; then
            size=$(du -sh "$model_dir" | cut -f1)
            echo "    Seed $seed: $size"
        fi
    done
    echo ""
fi

# ============================================================================
# 6. 快捷监控命令
# ============================================================================
echo "============================================================================"
echo "📝 快捷监控命令"
echo "============================================================================"
echo ""
echo "  查看特定训练日志 (例如 Seed 42):"
echo "    tail -f $BASE_OUTPUT_DIR/full_model_seed42/nohup.log"
echo ""
echo "  查看所有运行中的进程:"
if [ -f "$PID_FILE" ]; then
    mapfile -t PIDS < "$PID_FILE"
    echo "    ps -p ${PIDS[*]} -o pid,stat,etime,cmd"
fi
echo ""
echo "  查看GPU使用:"
echo "    nvidia-smi"
echo ""
echo "  实时监控此脚本:"
echo "    watch -n 60 ./check_full_model_progress.sh"
echo ""
echo "  终止所有训练:"
if [ -f "$PID_FILE" ]; then
    echo "    kill ${PIDS[*]}"
fi
echo ""
echo "  生成结果汇总:"
echo "    python summarize_full_model_results.py --model_dir $BASE_OUTPUT_DIR"
echo ""
echo "============================================================================"
echo ""
