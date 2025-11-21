#!/bin/bash

# ============================================================================
# 全模块（Full Model）多种子训练脚本 - 后台并行执行
# 运行3个随机种子的完整模型训练，所有任务在后台并行运行
# ============================================================================

# 基础配置
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
BASE_OUTPUT_DIR="./full_model_multi_seed"

# 训练超参数（与用户提供的完全一致）
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-4
WARMUP_STEPS=2000
ALIGNN_LAYERS=4
GCN_LAYERS=4
HIDDEN_FEATURES=256
GRAPH_DROPOUT=0.15
CROSS_MODAL_NUM_HEADS=4
MIDDLE_FUSION_LAYERS=2
FINE_GRAINED_HIDDEN_DIM=256
FINE_GRAINED_NUM_HEADS=8
FINE_GRAINED_DROPOUT=0.2
FINE_GRAINED_USE_PROJECTION=True
EARLY_STOPPING_PATIENCE=150
NUM_WORKERS=24

# 随机种子列表
SEEDS=(42 123 7)

# 公共参数
COMMON_ARGS="
    --root_dir $ROOT_DIR \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --alignn_layers $ALIGNN_LAYERS \
    --gcn_layers $GCN_LAYERS \
    --hidden_features $HIDDEN_FEATURES \
    --graph_dropout $GRAPH_DROPOUT \
    --cross_modal_num_heads $CROSS_MODAL_NUM_HEADS \
    --middle_fusion_layers $MIDDLE_FUSION_LAYERS \
    --fine_grained_hidden_dim $FINE_GRAINED_HIDDEN_DIM \
    --fine_grained_num_heads $FINE_GRAINED_NUM_HEADS \
    --fine_grained_dropout $FINE_GRAINED_DROPOUT \
    --fine_grained_use_projection $FINE_GRAINED_USE_PROJECTION \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --num_workers $NUM_WORKERS
"

# 创建基础输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 主日志文件
MAIN_LOG="$BASE_OUTPUT_DIR/launch_log_$(date +%Y%m%d_%H%M%S).txt"

# 用于存储后台进程的PID
declare -a ALL_PIDS
declare -a ALL_NAMES

echo "============================================================================" | tee -a "$MAIN_LOG"
echo "🚀 启动全模块训练（多种子版本 - 后台并行执行）" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "时间: $(date)" | tee -a "$MAIN_LOG"
echo "数据集: $DATASET/$PROPERTY" | tee -a "$MAIN_LOG"
echo "实验配置: Full Model × 3个种子 = 3个训练任务" | tee -a "$MAIN_LOG"
echo "执行模式: 后台并行" | tee -a "$MAIN_LOG"
echo "随机种子: ${SEEDS[@]}" | tee -a "$MAIN_LOG"
echo "基础输出目录: $BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# ============================================================================
# 启动所有Full Model训练任务
# ============================================================================

for seed in "${SEEDS[@]}"; do
    output_dir="$BASE_OUTPUT_DIR/full_model_seed${seed}"
    log_file="$output_dir/nohup.log"

    # 创建输出目录
    mkdir -p "$output_dir"

    echo "----------------------------------------" | tee -a "$MAIN_LOG"
    echo "启动: Full Model (seed=$seed)" | tee -a "$MAIN_LOG"
    echo "  输出目录: $output_dir" | tee -a "$MAIN_LOG"
    echo "  配置: 所有模块启用 (cross_modal=True, middle_fusion=True, fine_grained=True)" | tee -a "$MAIN_LOG"

    # 后台启动训练
    nohup python train_with_cross_modal_attention.py \
        $COMMON_ARGS \
        --random_seed $seed \
        --use_cross_modal True \
        --use_middle_fusion True \
        --use_fine_grained_attention True \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1 &

    pid=$!
    ALL_PIDS+=($pid)
    ALL_NAMES+=("FullModel_Seed${seed}")

    echo "  后台进程PID: $pid" | tee -a "$MAIN_LOG"
    echo "  日志文件: $log_file" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"

    # 短暂延迟
    sleep 2
done

# ============================================================================
# 启动完成汇总
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "✅ 所有Full Model训练已在后台启动！" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "总计: ${#ALL_PIDS[@]} 个训练任务" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "后台进程列表:" | tee -a "$MAIN_LOG"
echo "----------------------------------------" | tee -a "$MAIN_LOG"
for i in "${!ALL_PIDS[@]}"; do
    printf "  [%d] %s (PID: %d)\n" $((i+1)) "${ALL_NAMES[$i]}" "${ALL_PIDS[$i]}" | tee -a "$MAIN_LOG"
done
echo "" | tee -a "$MAIN_LOG"

# ============================================================================
# 监控命令提示
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📝 监控命令:" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "1. 查看所有进程状态:" | tee -a "$MAIN_LOG"
echo "   ps -p ${ALL_PIDS[*]} -o pid,stat,etime,cmd" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "2. 查看特定训练的日志 (例如 Seed 42):" | tee -a "$MAIN_LOG"
echo "   tail -f $BASE_OUTPUT_DIR/full_model_seed42/nohup.log" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "3. 查看所有训练的最新状态:" | tee -a "$MAIN_LOG"
echo "   ./check_full_model_progress.sh" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "4. 终止所有训练:" | tee -a "$MAIN_LOG"
echo "   kill ${ALL_PIDS[*]}" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "5. 查看GPU使用情况:" | tee -a "$MAIN_LOG"
echo "   nvidia-smi" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "6. 生成结果汇总:" | tee -a "$MAIN_LOG"
echo "   python summarize_full_model_results.py --model_dir $BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "主日志文件: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 保存PID列表到文件，方便后续管理
PID_FILE="$BASE_OUTPUT_DIR/running_pids.txt"
printf "%s\n" "${ALL_PIDS[@]}" > "$PID_FILE"
echo "✅ 进程PID已保存到: $PID_FILE" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
