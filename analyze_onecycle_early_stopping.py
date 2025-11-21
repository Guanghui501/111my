#!/usr/bin/env python
"""
分析OneCycleLR与Early Stopping的交互
比较：完整训练 vs early stopping
"""

import json
import os
import sys

def analyze_training_history(output_dir):
    """分析训练历史，找出early stopping的影响"""

    history_val_file = os.path.join(output_dir, 'history_val.json')
    history_train_file = os.path.join(output_dir, 'history_train.json')

    if not os.path.exists(history_val_file):
        print(f"错误: 找不到 {history_val_file}")
        return

    with open(history_val_file, 'r') as f:
        val_history = json.load(f)

    with open(history_train_file, 'r') as f:
        train_history = json.load(f)

    epochs = val_history['epochs']

    # 检测是回归还是分类
    if 'mae' in val_history:
        metric_name = 'mae'
        is_lower_better = True
    elif 'accuracy' in val_history:
        metric_name = 'accuracy'
        is_lower_better = False
    else:
        print("无法确定任务类型")
        return

    val_metrics = val_history[metric_name]
    train_loss = train_history['loss']

    print("="*80)
    print("OneCycleLR 与 Early Stopping 交互分析")
    print("="*80)

    # 找到最佳验证性能及其位置
    if is_lower_better:
        best_val = min(val_metrics)
        best_epoch = val_metrics.index(best_val)
    else:
        best_val = max(val_metrics)
        best_epoch = val_metrics.index(best_val)

    total_epochs = len(epochs)
    last_epoch = epochs[-1]

    print(f"\n训练概况:")
    print(f"  总训练轮数: {total_epochs}")
    print(f"  最后epoch编号: {last_epoch}")
    print(f"  最佳验证{metric_name}: {best_val:.4f} (出现在 epoch {epochs[best_epoch]})")
    print(f"  最后验证{metric_name}: {val_metrics[-1]:.4f} (epoch {last_epoch})")
    print(f"  未改善轮数: {last_epoch - epochs[best_epoch]}")

    # 分析训练的不同阶段
    print(f"\n训练阶段分析:")

    # 前25%
    quarter_1 = total_epochs // 4
    if quarter_1 > 0:
        val_q1 = val_metrics[quarter_1-1]
        print(f"  前25% (epoch {epochs[quarter_1-1]}): Val {metric_name} = {val_q1:.4f}")

    # 50%
    half = total_epochs // 2
    if half > 0:
        val_half = val_metrics[half-1]
        print(f"  50% (epoch {epochs[half-1]}): Val {metric_name} = {val_half:.4f}")

    # 75%
    quarter_3 = (total_epochs * 3) // 4
    if quarter_3 > 0:
        val_q3 = val_metrics[quarter_3-1]
        print(f"  75% (epoch {epochs[quarter_3-1]}): Val {metric_name} = {val_q3:.4f}")

    print(f"  100% (epoch {last_epoch}): Val {metric_name} = {val_metrics[-1]:.4f}")

    # 分析最后阶段的趋势
    print(f"\n后期训练趋势分析:")

    if total_epochs >= 100:
        # 最后100轮的趋势
        last_100_vals = val_metrics[-100:]
        last_100_epochs = epochs[-100:]

        if is_lower_better:
            improving = sum(1 for i in range(1, len(last_100_vals))
                          if last_100_vals[i] < last_100_vals[i-1])
        else:
            improving = sum(1 for i in range(1, len(last_100_vals))
                          if last_100_vals[i] > last_100_vals[i-1])

        print(f"  最后100轮中改善的轮数: {improving}/99")
        print(f"  最后100轮验证{metric_name}范围: {min(last_100_vals):.4f} - {max(last_100_vals):.4f}")

        # 检查是否还在下降/上升趋势中
        last_20_vals = val_metrics[-20:]
        first_10_avg = sum(last_20_vals[:10]) / 10
        last_10_avg = sum(last_20_vals[-10:]) / 10

        if is_lower_better:
            trend = "改善" if last_10_avg < first_10_avg else "恶化"
            trend_val = first_10_avg - last_10_avg
        else:
            trend = "改善" if last_10_avg > first_10_avg else "恶化"
            trend_val = last_10_avg - first_10_avg

        print(f"  最后20轮趋势: {trend} ({abs(trend_val):.4f})")

    # 过拟合分析
    print(f"\n过拟合分析:")
    train_loss_best = train_loss[best_epoch]
    train_loss_last = train_loss[-1]
    val_best = val_metrics[best_epoch]
    val_last = val_metrics[-1]

    print(f"  最佳验证性能时 (epoch {epochs[best_epoch]}):")
    print(f"    训练Loss: {train_loss_best:.4f}")
    print(f"    验证{metric_name}: {val_best:.4f}")

    print(f"  训练结束时 (epoch {last_epoch}):")
    print(f"    训练Loss: {train_loss_last:.4f}")
    print(f"    验证{metric_name}: {val_last:.4f}")

    if train_loss_last < train_loss_best:
        overfitting_indicator = abs(val_last - val_best)
        print(f"  ⚠️  训练Loss持续降低，但验证性能恶化 {overfitting_indicator:.4f} → 过拟合")

    # OneCycleLR建议
    print(f"\n" + "="*80)
    print("OneCycleLR 使用建议:")
    print("="*80)

    completion_rate = total_epochs / 1000.0 if 'epochs' in val_history else 0

    if last_epoch < 900:  # 假设目标是1000 epochs
        print(f"""
⚠️  训练可能提前终止了！

当前状态:
  - 计划训练轮数: 可能是1000
  - 实际训练轮数: {total_epochs}
  - Early stopping触发: epoch {last_epoch}

OneCycleLR特性:
  1. 学习率周期: 上升 → 高位 → 下降
  2. 后期低学习率阶段(约epoch 700-1000)对精细调优很重要
  3. 提前停止会错过低学习率精细调优阶段

建议:

  选项1: 增加early stopping耐心值
    "n_early_stopping": 500  # 从150增加到500

  选项2: 禁用early stopping（推荐用于OneCycleLR）
    "n_early_stopping": null

  选项3: 如果使用early stopping，调整epochs匹配预期停止点
    "epochs": {last_epoch + 100}  # 设置为预期停止点

  选项4: 改用其他调度器（如果必须用early stopping）
    "scheduler": "lambda"  # 或 "step"
        """)
    else:
        print(f"""
✓ 训练完成了较完整的周期

建议:
  - 当前配置合理
  - 验证性能: {val_last:.4f}
  - 如果需要进一步改进，可以尝试更多epoch
        """)

    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_onecycle_early_stopping.py <output_dir>")
        print("示例: python analyze_onecycle_early_stopping.py ./output_1000epochs_bs128_sw_ju/mbj_bandgap/")
        sys.exit(1)

    output_dir = sys.argv[1]
    analyze_training_history(output_dir)
