#!/bin/bash

# 设置日志文件
LOG_FILE="./classification_training_$(date +%Y%m%d_%H%M%S).log"

# 分类训练命令
nohup python train_with_cross_modal_attention.py \
    --root_dir ./dataset \
    --dataset class \
    --property syn \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 128 \
    --epochs 1000 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection True \
    --classification 1 \
    --early_stopping_patience 150 \
    --output_dir ./output_classification_1000epochs_bs128 \
    --num_workers 12 \
    --random_seed 42 \
    > "$LOG_FILE" 2>&1 &

echo "分类训练已启动，日志文件: $LOG_FILE"
echo "使用命令查看训练进度: tail -f $LOG_FILE"
echo "查看后台任务: jobs"
