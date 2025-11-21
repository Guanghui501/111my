#!/usr/bin/env python
"""
消融实验可视化脚本
生成各种对比图表
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_style("white")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_performance_comparison(df, save_dir, metric='mae'):
    """绘制性能对比柱状图"""

    # 准备数据
    experiments = df['Experiment'].values
    if metric == 'mae':
        values = df['test_mae'].values
        ylabel = 'MAE (eV)'
        title = 'Ablation Study: Test MAE Comparison'
        better = 'lower'
    elif metric == 'accuracy':
        values = df['test_accuracy'].values
        ylabel = 'Accuracy'
        title = 'Ablation Study: Test Accuracy Comparison'
        better = 'higher'
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 颜色映射
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2']

    bars = ax.bar(range(len(experiments)), values, color=colors, edgecolor='black', linewidth=1.5)

    # 设置x轴标签
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([exp.replace('Exp-', '').replace(': ', '\n') for exp in experiments],
                       rotation=0, ha='center')

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16, pad=20)

    # 添加最佳性能虚线
    if better == 'lower':
        best_value = min(values)
        ax.axhline(y=best_value, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best: {best_value:.4f}')
    else:
        best_value = max(values)
        ax.axhline(y=best_value, color='green', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best: {best_value:.4f}')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend(loc='best')

    plt.tight_layout()
    output_path = os.path.join(save_dir, f'ablation_comparison_{metric}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存对比图: {output_path}")
    plt.close()


def plot_improvement_analysis(df, save_dir, metric='mae'):
    """绘制相对提升分析图"""

    experiments = df['Experiment'].values
    if metric == 'mae':
        values = df['test_mae'].values
        baseline = values[0]
        improvements = (baseline - values) / baseline * 100
        ylabel = 'Relative Improvement (%)'
        title = 'Relative MAE Improvement from Baseline'
    elif metric == 'accuracy':
        values = df['test_accuracy'].values
        baseline = values[0]
        improvements = (values - baseline) * 100
        ylabel = 'Absolute Improvement (%)'
        title = 'Absolute Accuracy Improvement from Baseline'
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['gray', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2']
    bars = ax.bar(range(len(experiments)), improvements, color=colors,
                  edgecolor='black', linewidth=1.5)

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([exp.replace('Exp-', '').replace(': ', '\n') for exp in experiments],
                       rotation=0, ha='center')

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16, pad=20)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # 标注数值
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.2f}%',
               ha='center', va='bottom' if val > 0 else 'top',
               fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(save_dir, f'ablation_improvement_{metric}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存提升分析图: {output_path}")
    plt.close()


def plot_component_contribution(df, save_dir, metric='mae'):
    """绘制组件贡献分析图 - 突出你的两个创新"""

    # 关键对比
    comparisons = []

    # 找到关键实验
    exp3 = df[df['Experiment'].str.contains('Cross-Modal') &
              ~df['Experiment'].str.contains('Middle') &
              ~df['Experiment'].str.contains('Fine')].iloc[0] if len(df) >= 3 else None
    exp4 = df[df['Experiment'].str.contains('Middle')].iloc[0] if 'Middle' in str(df['Experiment'].values) else None
    exp5 = df[df['Experiment'].str.contains('Fine')].iloc[0] if 'Fine' in str(df['Experiment'].values) else None
    exp6 = df[df['Experiment'].str.contains('Full')].iloc[0] if 'Full' in str(df['Experiment'].values) else None

    if metric == 'mae':
        metric_col = 'test_mae'
        ylabel = 'MAE (eV)'
        title = 'Component Contribution Analysis'
        better = 'lower'
    else:
        metric_col = 'test_accuracy'
        ylabel = 'Accuracy'
        title = 'Component Contribution Analysis'
        better = 'higher'

    # 准备数据
    labels = []
    values = []

    if exp3 is not None:
        labels.append('Baseline\n(Cross-Modal)')
        values.append(exp3[metric_col])

    if exp4 is not None:
        labels.append('+Middle\nFusion')
        values.append(exp4[metric_col])

    if exp5 is not None:
        labels.append('+Fine-Grained\nAttention')
        values.append(exp5[metric_col])

    if exp6 is not None:
        labels.append('Full Model\n(Both)')
        values.append(exp6[metric_col])

    if len(values) < 2:
        print("⚠️  数据不足，跳过组件贡献分析图")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#e377c2']
    bars = ax.bar(range(len(labels)), values, color=colors[:len(values)],
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16, pad=20)

    # 标注数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加改进箭头和百分比
    if better == 'lower':
        for i in range(len(values) - 1):
            improvement = (values[i] - values[i+1]) / values[i] * 100
            mid_x = i + 0.5
            mid_y = (values[i] + values[i+1]) / 2
            ax.annotate(f'{improvement:.1f}%↓',
                       xy=(mid_x, mid_y), fontsize=10, color='red',
                       ha='center', fontweight='bold')
    else:
        for i in range(len(values) - 1):
            improvement = (values[i+1] - values[i]) / values[i] * 100
            mid_x = i + 0.5
            mid_y = (values[i] + values[i+1]) / 2
            ax.annotate(f'{improvement:.1f}%↑',
                       xy=(mid_x, mid_y), fontsize=10, color='green',
                       ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(save_dir, f'ablation_component_contribution_{metric}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存组件贡献图: {output_path}")
    plt.close()


def plot_training_curves_comparison(ablation_dir, save_dir):
    """对比不同配置的训练曲线"""

    exp_dirs = [
        ('exp3_cross_modal', 'Cross-Modal Only', '#2ca02c'),
        ('exp4_middle_fusion', '+Middle Fusion', '#1f77b4'),
        ('exp5_fine_grained', '+Fine-Grained', '#9467bd'),
        ('exp6_full_model', 'Full Model', '#e377c2'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for exp_id, label, color in exp_dirs:
        exp_path = os.path.join(ablation_dir, exp_id)
        history_val = os.path.join(exp_path, 'history_val.json')

        if not os.path.exists(history_val):
            continue

        import json
        with open(history_val, 'r') as f:
            history = json.load(f)

        epochs = list(range(1, len(history['loss']) + 1))

        # Loss曲线
        ax1.plot(epochs, history['loss'], label=label, color=color, linewidth=2, alpha=0.8)

        # MAE或Accuracy曲线
        if 'mae' in history:
            ax2.plot(epochs, history['mae'], label=label, color=color, linewidth=2, alpha=0.8)
        elif 'accuracy' in history:
            ax2.plot(epochs, history['accuracy'], label=label, color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    if 'mae' in history:
        ax2.set_ylabel('Validation MAE')
        ax2.set_title('Validation MAE Comparison')
    else:
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(save_dir, 'ablation_training_curves.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存训练曲线对比图: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='消融实验可视化')
    parser.add_argument('--ablation_dir', type=str, required=True,
                        help='消融实验根目录')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='图片保存目录（默认为ablation_dir/figures）')

    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(args.ablation_dir, 'figures')

    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("📊 消融实验可视化")
    print("="*80)
    print(f"消融实验目录: {args.ablation_dir}")
    print(f"保存目录: {args.save_dir}")
    print()

    # 读取汇总结果
    summary_file = os.path.join(args.ablation_dir, 'ablation_summary.csv')
    if not os.path.exists(summary_file):
        print(f"❌ 未找到汇总文件: {summary_file}")
        print("请先运行: python summarize_ablation_results.py --ablation_dir {ablation_dir}")
        return

    df = pd.read_csv(summary_file)
    print(f"✅ 加载汇总结果: {len(df)} 个实验")

    # 检测任务类型
    if 'test_mae' in df.columns:
        metric = 'mae'
        print("任务类型: 回归")
    elif 'test_accuracy' in df.columns:
        metric = 'accuracy'
        print("任务类型: 分类")
    else:
        print("❌ 无法识别任务类型")
        return

    print("\n生成可视化图表...")

    # 1. 性能对比图
    plot_performance_comparison(df, args.save_dir, metric)

    # 2. 相对提升分析图
    plot_improvement_analysis(df, args.save_dir, metric)

    # 3. 组件贡献分析图
    plot_component_contribution(df, args.save_dir, metric)

    # 4. 训练曲线对比
    try:
        plot_training_curves_comparison(args.ablation_dir, args.save_dir)
    except Exception as e:
        print(f"⚠️  训练曲线对比图生成失败: {e}")

    print("\n" + "="*80)
    print("✅ 所有图表生成完成！")
    print("="*80)
    print(f"保存位置: {args.save_dir}")
    print("\n生成的文件:")
    for f in os.listdir(args.save_dir):
        if f.startswith('ablation_'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
