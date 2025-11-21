#!/bin/bash

# ============================================================================
# 消融实验自动化脚本
# 测试中期融合和细粒度对齐的有效性
# ============================================================================

# 实验配置
DATASET="jarvis"
PROPERTY="mbj_bandgap"
ROOT_DIR="./dataset"
EPOCHS=300
BATCH_SIZE=128
LEARNING_RATE=1e-3
BASE_OUTPUT_DIR="./ablation_experiments"
RANDOM_SEED=42

# 公共参数
COMMON_ARGS="
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 5e-4 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --early_stopping_patience 50 \
    --num_workers 8 \
    --random_seed $RANDOM_SEED
"

# 创建实验目录
mkdir -p "$BASE_OUTPUT_DIR"

# 日志文件
LOG_FILE="$BASE_OUTPUT_DIR/ablation_log_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================================================" | tee -a "$LOG_FILE"
echo "消融实验开始" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"
echo "数据集: $DATASET/$PROPERTY" | tee -a "$LOG_FILE"
echo "基础输出目录: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Exp-1: ALIGNN Only (基线)
# 不使用任何跨模态组件
# ============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "[1/6] 运行 Exp-1: ALIGNN Only (基线)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
EXP1_DIR="$BASE_OUTPUT_DIR/exp1_alignn_only"

# 注意：这里需要运行不带文本的版本，或者简单拼接
# 如果你的代码支持--no_text参数，用它；否则设置所有跨模态为False
python train_with_cross_modal_attention.py \
    $COMMON_ARGS \
    --use_cross_modal False \
    --use_middle_fusion False \
    --use_fine_grained_attention False \
    --output_dir "$EXP1_DIR" \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Exp-1 完成" | tee -a "$LOG_FILE"
else
    echo "❌ Exp-1 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Exp-2: +Text (Simple Concatenation)
# 使用文本但不用跨模态注意力
# ============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "[2/6] 运行 Exp-2: +Text (Simple Concat)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
EXP2_DIR="$BASE_OUTPUT_DIR/exp2_text_concat"

python train_with_cross_modal_attention.py \
    $COMMON_ARGS \
    --use_cross_modal False \
    --use_middle_fusion False \
    --use_fine_grained_attention False \
    --output_dir "$EXP2_DIR" \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Exp-2 完成" | tee -a "$LOG_FILE"
else
    echo "❌ Exp-2 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Exp-3: +Cross-Modal Attention (后期融合)
# 仅使用跨模态注意力，不用中期融合和细粒度对齐
# ============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "[3/6] 运行 Exp-3: +Cross-Modal (后期融合)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
EXP3_DIR="$BASE_OUTPUT_DIR/exp3_cross_modal"

python train_with_cross_modal_attention.py \
    $COMMON_ARGS \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion False \
    --use_fine_grained_attention False \
    --output_dir "$EXP3_DIR" \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Exp-3 完成" | tee -a "$LOG_FILE"
else
    echo "❌ Exp-3 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Exp-4: +Middle Fusion (你的创新1)
# 跨模态 + 中期融合，但不用细粒度对齐
# ============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "[4/6] 运行 Exp-4: +Middle Fusion (创新1)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
EXP4_DIR="$BASE_OUTPUT_DIR/exp4_middle_fusion"

python train_with_cross_modal_attention.py \
    $COMMON_ARGS \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --output_dir "$EXP4_DIR" \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Exp-4 完成" | tee -a "$LOG_FILE"
else
    echo "❌ Exp-4 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Exp-5: +Fine-Grained Attention (你的创新2)
# 跨模态 + 细粒度对齐，但不用中期融合
# ============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "[5/6] 运行 Exp-5: +Fine-Grained (创新2)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
EXP5_DIR="$BASE_OUTPUT_DIR/exp5_fine_grained"

python train_with_cross_modal_attention.py \
    $COMMON_ARGS \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion False \
    --use_fine_grained_attention True \
    --fine_grained_num_heads 8 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_use_projection True \
    --output_dir "$EXP5_DIR" \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Exp-5 完成" | tee -a "$LOG_FILE"
else
    echo "❌ Exp-5 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Exp-6: Full Model (完整模型)
# 跨模态 + 中期融合 + 细粒度对齐
# ============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "[6/6] 运行 Exp-6: Full Model (完整模型)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
EXP6_DIR="$BASE_OUTPUT_DIR/exp6_full_model"

python train_with_cross_modal_attention.py \
    $COMMON_ARGS \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --fine_grained_num_heads 8 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_use_projection True \
    --output_dir "$EXP6_DIR" \
    >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Exp-6 完成" | tee -a "$LOG_FILE"
else
    echo "❌ Exp-6 失败" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# 汇总结果
# ============================================================================
echo "============================================================================" | tee -a "$LOG_FILE"
echo "所有实验完成！" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "正在生成结果汇总..." | tee -a "$LOG_FILE"
python summarize_ablation_results.py --ablation_dir "$BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ 消融实验全部完成！" | tee -a "$LOG_FILE"
echo "查看日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "查看汇总: $BASE_OUTPUT_DIR/ablation_summary.csv" | tee -a "$LOG_FILE"
