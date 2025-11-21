#!/bin/bash

# ============================================================================
# 消融实验自动化脚本（多种子版本）
# 运行4个实验配置 × 3个随机种子 = 12个训练任务
# ============================================================================

# 基础配置
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
BASE_OUTPUT_DIR="./ablation_multi_seed"

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

echo "============================================================================" | tee -a "$MAIN_LOG"
echo "🚀 启动消融实验（多种子版本）" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "时间: $(date)" | tee -a "$MAIN_LOG"
echo "数据集: $DATASET/$PROPERTY" | tee -a "$MAIN_LOG"
echo "实验配置: 4个实验 × 3个种子 = 12个训练任务" | tee -a "$MAIN_LOG"
echo "随机种子: ${SEEDS[@]}" | tee -a "$MAIN_LOG"
echo "基础输出目录: $BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 用于存储所有后台进程的PID
declare -a ALL_PIDS
declare -a ALL_NAMES

# ============================================================================
# 实验函数：启动单个训练任务
# ============================================================================
launch_experiment() {
    local exp_name=$1
    local exp_num=$2
    local seed=$3
    local use_cross_modal=$4
    local use_middle_fusion=$5
    local use_fine_grained=$6

    local output_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"
    local log_file="$output_dir/nohup.log"

    # 创建输出目录
    mkdir -p "$output_dir"

    echo "----------------------------------------" | tee -a "$MAIN_LOG"
    echo "启动: $exp_name (seed=$seed)" | tee -a "$MAIN_LOG"
    echo "  输出目录: $output_dir" | tee -a "$MAIN_LOG"
    echo "  配置: cross_modal=$use_cross_modal, middle_fusion=$use_middle_fusion, fine_grained=$use_fine_grained" | tee -a "$MAIN_LOG"

    # 启动后台训练
    nohup python train_with_cross_modal_attention.py \
        $COMMON_ARGS \
        --random_seed $seed \
        --use_cross_modal $use_cross_modal \
        --use_middle_fusion $use_middle_fusion \
        --use_fine_grained_attention $use_fine_grained \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1 &

    local pid=$!
    ALL_PIDS+=($pid)
    ALL_NAMES+=("Exp${exp_num}_Seed${seed}")

    echo "  后台进程PID: $pid" | tee -a "$MAIN_LOG"
    echo "  日志文件: $log_file" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"

    # 短暂延迟，避免同时启动过多进程
    sleep 2
}

# ============================================================================
# 实验1: Text Simple Concat (Baseline)
# 不使用任何跨模态注意力机制
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📊 实验1: Text Simple Concat (Baseline)" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"

for seed in "${SEEDS[@]}"; do
    launch_experiment \
        "Exp-1: Baseline" \
        1 \
        $seed \
        False \
        False \
        False
done

# ============================================================================
# 实验2: +Late Fusion
# 添加晚期跨模态注意力（全局级别融合）
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📊 实验2: +Late Fusion" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"

for seed in "${SEEDS[@]}"; do
    launch_experiment \
        "Exp-2: +Late Fusion" \
        2 \
        $seed \
        True \
        False \
        False
done

# ============================================================================
# 实验3: +Late Fusion +Middle Fusion (创新1)
# Late Fusion + 中期融合（在编码过程中注入文本信息）
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📊 实验3: +Late Fusion +Middle Fusion (创新1)" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"

for seed in "${SEEDS[@]}"; do
    launch_experiment \
        "Exp-3: +Middle Fusion" \
        3 \
        $seed \
        True \
        True \
        False
done

# ============================================================================
# 实验4: +Late Fusion +Fine-Grained (创新2)
# Late Fusion + 细粒度注意力（原子-词级别对齐）
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📊 实验4: +Late Fusion +Fine-Grained (创新2)" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"

for seed in "${SEEDS[@]}"; do
    launch_experiment \
        "Exp-4: +Fine-Grained" \
        4 \
        $seed \
        True \
        False \
        True
done

# ============================================================================
# 启动完成汇总
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "✅ 所有实验已在后台启动！" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "总计: ${#ALL_PIDS[@]} 个训练任务" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "后台进程列表:" | tee -a "$MAIN_LOG"
echo "----------------------------------------" | tee -a "$MAIN_LOG"
for i in "${!ALL_PIDS[@]}"; do
    printf "  [%2d] %s (PID: %d)\n" $((i+1)) "${ALL_NAMES[$i]}" "${ALL_PIDS[$i]}" | tee -a "$MAIN_LOG"
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
echo "2. 查看特定实验的日志 (例如 Exp1, Seed42):" | tee -a "$MAIN_LOG"
echo "   tail -f $BASE_OUTPUT_DIR/exp1_seed42/nohup.log" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "3. 查看所有实验的最新状态:" | tee -a "$MAIN_LOG"
echo "   ./check_ablation_multi_seed_progress.sh" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "4. 终止所有实验:" | tee -a "$MAIN_LOG"
echo "   kill ${ALL_PIDS[*]}" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "5. 查看GPU使用情况:" | tee -a "$MAIN_LOG"
echo "   nvidia-smi" | tee -a "$MAIN_LOG"
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
