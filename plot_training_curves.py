#!/usr/bin/env python
"""
绘制训练历史曲线
从 history_train.json 和 history_val.json 生成训练过程可视化
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18


def plot_training_history(output_dir, save_dir=None, show=True):
    """
    绘制训练历史曲线

    Args:
        output_dir: 包含history_*.json文件的目录
        save_dir: 保存图片的目录（如果None则保存到output_dir）
        show: 是否显示图片
    """
    if save_dir is None:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # 加载训练和验证历史
    train_history_file = os.path.join(output_dir, 'history_train.json')
    val_history_file = os.path.join(output_dir, 'history_val.json')

    if not os.path.exists(train_history_file):
        print(f"❌ 未找到训练历史文件: {train_history_file}")
        return
    if not os.path.exists(val_history_file):
        print(f"❌ 未找到验证历史文件: {val_history_file}")
        return

    with open(train_history_file, 'r') as f:
        train_history = json.load(f)
    with open(val_history_file, 'r') as f:
        val_history = json.load(f)

    print(f"✅ 加载历史数据成功")
    print(f"   训练指标: {list(train_history.keys())}")
    print(f"   验证指标: {list(val_history.keys())}")

    # 检测任务类型（分类或回归）
    metrics_list = list(train_history.keys())
    is_classification = 'accuracy' in metrics_list

    if is_classification:
        print(f"📊 检测到分类任务")
        main_metric = 'accuracy'
        metric_label = 'Accuracy'
        metric_better = 'higher'
    else:
        print(f"📊 检测到回归任务")
        main_metric = 'mae'
        metric_label = 'MAE'
        metric_better = 'lower'

    epochs = list(range(1, len(train_history['loss']) + 1))

    # ========== 图1: Loss曲线 ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_history['loss'],
            label='Training Loss', linewidth=2, marker='o', markersize=4,
            markevery=max(1, len(epochs)//20))
    ax.plot(epochs, val_history['loss'],
            label='Validation Loss', linewidth=2, marker='s', markersize=4,
            markevery=max(1, len(epochs)//20))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注最小验证损失
    min_val_loss_epoch = np.argmin(val_history['loss']) + 1
    min_val_loss = min(val_history['loss'])
    ax.axvline(x=min_val_loss_epoch, color='red', linestyle='--',
               alpha=0.5, label=f'Best Val Loss (Epoch {min_val_loss_epoch})')
    ax.legend()

    plt.tight_layout()
    loss_fig_path = os.path.join(save_dir, 'training_loss_curve.pdf')
    plt.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(loss_fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存Loss曲线: {loss_fig_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 图2: 主要指标曲线 (MAE或Accuracy) ==========
    if main_metric in train_history:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, train_history[main_metric],
                label=f'Training {metric_label}', linewidth=2, marker='o',
                markersize=4, markevery=max(1, len(epochs)//20))
        ax.plot(epochs, val_history[main_metric],
                label=f'Validation {metric_label}', linewidth=2, marker='s',
                markersize=4, markevery=max(1, len(epochs)//20))

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_label)
        ax.set_title(f'Training and Validation {metric_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 标注最佳指标
        if metric_better == 'lower':
            best_val_epoch = np.argmin(val_history[main_metric]) + 1
            best_val_metric = min(val_history[main_metric])
        else:
            best_val_epoch = np.argmax(val_history[main_metric]) + 1
            best_val_metric = max(val_history[main_metric])

        ax.axvline(x=best_val_epoch, color='red', linestyle='--',
                   alpha=0.5, label=f'Best Val (Epoch {best_val_epoch})')
        ax.legend()

        plt.tight_layout()
        metric_fig_path = os.path.join(save_dir, f'training_{main_metric}_curve.pdf')
        plt.savefig(metric_fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(metric_fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f"✅ 保存{metric_label}曲线: {metric_fig_path}")

        if show:
            plt.show()
        plt.close()

    # ========== 图3: 分类任务额外指标 (Precision, Recall) ==========
    if is_classification and 'precision' in train_history and 'recall' in train_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Precision
        ax1.plot(epochs, train_history['precision'],
                label='Training Precision', linewidth=2, marker='o', markersize=4,
                markevery=max(1, len(epochs)//20))
        ax1.plot(epochs, val_history['precision'],
                label='Validation Precision', linewidth=2, marker='s', markersize=4,
                markevery=max(1, len(epochs)//20))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Precision')
        ax1.set_title('Training and Validation Precision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Recall
        ax2.plot(epochs, train_history['recall'],
                label='Training Recall', linewidth=2, marker='o', markersize=4,
                markevery=max(1, len(epochs)//20))
        ax2.plot(epochs, val_history['recall'],
                label='Validation Recall', linewidth=2, marker='s', markersize=4,
                markevery=max(1, len(epochs)//20))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Recall')
        ax2.set_title('Training and Validation Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        class_metrics_path = os.path.join(save_dir, 'training_classification_metrics.pdf')
        plt.savefig(class_metrics_path, dpi=300, bbox_inches='tight')
        plt.savefig(class_metrics_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f"✅ 保存分类指标曲线: {class_metrics_path}")

        if show:
            plt.show()
        plt.close()

    # ========== 图4: 综合对比图 ==========
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 上图：Loss
    axes[0].plot(epochs, train_history['loss'],
                label='Training Loss', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, val_history['loss'],
                label='Validation Loss', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：主要指标
    if main_metric in train_history:
        axes[1].plot(epochs, train_history[main_metric],
                    label=f'Training {metric_label}', linewidth=2, alpha=0.8)
        axes[1].plot(epochs, val_history[main_metric],
                    label=f'Validation {metric_label}', linewidth=2, alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_label)
        axes[1].set_title(f'{metric_label} Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = os.path.join(save_dir, 'training_curves_combined.pdf')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.savefig(combined_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存综合曲线: {combined_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 打印统计信息 ==========
    print("\n" + "="*60)
    print("📊 训练统计信息")
    print("="*60)
    print(f"总训练轮次: {len(epochs)}")
    print(f"最终训练Loss: {train_history['loss'][-1]:.6f}")
    print(f"最终验证Loss: {val_history['loss'][-1]:.6f}")
    print(f"最小验证Loss: {min_val_loss:.6f} (Epoch {min_val_loss_epoch})")

    if main_metric in train_history:
        print(f"\n最终训练{metric_label}: {train_history[main_metric][-1]:.6f}")
        print(f"最终验证{metric_label}: {val_history[main_metric][-1]:.6f}")
        print(f"最佳验证{metric_label}: {best_val_metric:.6f} (Epoch {best_val_epoch})")

    if is_classification:
        if 'precision' in val_history and 'recall' in val_history:
            print(f"\n最终验证Precision: {val_history['precision'][-1]:.6f}")
            print(f"最终验证Recall: {val_history['recall'][-1]:.6f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制训练历史曲线')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='包含history_*.json文件的输出目录')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存图片的目录（默认与output_dir相同）')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图片（仅保存）')

    args = parser.parse_args()

    print("="*60)
    print("📈 开始绘制训练历史曲线")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    print(f"保存目录: {args.save_dir or args.output_dir}")
    print()

    plot_training_history(
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        show=not args.no_show
    )

    print("✅ 所有图表生成完成！")
