#!/usr/bin/env python
"""
绘制预测结果图
从 predictions_*.csv 生成预测vs真实值的散点图
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18


def calculate_regression_metrics(y_true, y_pred):
    """计算回归指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """计算分类指标"""
    # 确保标签是整数
    y_true = y_true.astype(int)
    y_pred_binary = (y_pred > 0.5).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred_binary),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'F1': f1_score(y_true, y_pred_binary, zero_division=0)
    }

    # 如果提供了概率预测，计算AUC
    if y_pred_proba is not None or len(np.unique(y_pred)) > 2:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_pred)
        except:
            pass

    return metrics


def plot_regression_predictions(output_dir, save_dir=None, show=True):
    """
    绘制回归预测结果

    Args:
        output_dir: 包含predictions_*.csv文件的目录
        save_dir: 保存图片的目录（如果None则保存到output_dir）
        show: 是否显示图片
    """
    if save_dir is None:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # 查找预测文件
    pred_files = {
        'train': os.path.join(output_dir, 'predictions_train.csv'),
        'val': os.path.join(output_dir, 'predictions_val.csv'),
        'test': os.path.join(output_dir, 'predictions_test.csv')
    }

    # 检查哪些文件存在
    available_sets = {}
    for set_name, file_path in pred_files.items():
        if os.path.exists(file_path):
            available_sets[set_name] = file_path
            print(f"✅ 找到{set_name}集预测文件: {file_path}")
        else:
            print(f"⚠️  未找到{set_name}集预测文件: {file_path}")

    if not available_sets:
        print("❌ 未找到任何预测文件！")
        return

    # 读取数据
    data = {}
    for set_name, file_path in available_sets.items():
        df = pd.read_csv(file_path)
        print(f"   {set_name}集样本数: {len(df)}")
        data[set_name] = df

    # 检测是分类还是回归任务
    # 通过检查预测值的唯一值数量来判断
    first_df = list(data.values())[0]
    unique_targets = first_df['target'].nunique()

    is_classification = unique_targets <= 10 and set(first_df['target'].unique()).issubset({0, 1})

    if is_classification:
        print("\n📊 检测到分类任务")
        plot_classification_predictions(data, save_dir, show)
    else:
        print("\n📊 检测到回归任务")
        plot_regression_scatter(data, save_dir, show)


def plot_regression_scatter(data, save_dir, show):
    """绘制回归任务的散点图"""

    n_sets = len(data)

    # ========== 图1: 三个子图分别显示 ==========
    fig, axes = plt.subplots(1, n_sets, figsize=(6*n_sets, 5))
    if n_sets == 1:
        axes = [axes]

    metrics_summary = {}

    for idx, (set_name, df) in enumerate(data.items()):
        ax = axes[idx]

        y_true = df['target'].values
        y_pred = df['prediction'].values

        # 计算指标
        metrics = calculate_regression_metrics(y_true, y_pred)
        metrics_summary[set_name] = metrics

        # 绘制散点图
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)

        # 添加对角线 (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')

        # 设置标签
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{set_name.capitalize()} Set')

        # 添加指标文本
        textstr = f"MAE = {metrics['MAE']:.4f}\nRMSE = {metrics['RMSE']:.4f}\nR² = {metrics['R²']:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    separate_path = os.path.join(save_dir, 'predictions_separate.pdf')
    plt.savefig(separate_path, dpi=300, bbox_inches='tight')
    plt.savefig(separate_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存分开的预测图: {separate_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 图2: 合并在一个图中 ==========
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # train, val, test
    markers = ['o', 's', '^']

    for idx, (set_name, df) in enumerate(data.items()):
        y_true = df['target'].values
        y_pred = df['prediction'].values

        metrics = metrics_summary[set_name]
        label = f"{set_name.capitalize()}: MAE={metrics['MAE']:.4f}, R²={metrics['R²']:.3f}"

        ax.scatter(y_true, y_pred, alpha=0.5, s=40,
                  color=colors[idx], marker=markers[idx],
                  edgecolors='k', linewidth=0.5, label=label)

    # 添加对角线
    all_true = np.concatenate([df['target'].values for df in data.values()])
    all_pred = np.concatenate([df['prediction'].values for df in data.values()])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='y=x')

    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Predictions vs. True Values (All Sets)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = os.path.join(save_dir, 'predictions_combined.pdf')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.savefig(combined_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存合并的预测图: {combined_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 图3: 残差图 ==========
    fig, axes = plt.subplots(1, n_sets, figsize=(6*n_sets, 5))
    if n_sets == 1:
        axes = [axes]

    for idx, (set_name, df) in enumerate(data.items()):
        ax = axes[idx]

        y_true = df['target'].values
        y_pred = df['prediction'].values
        residuals = y_true - y_pred

        # 残差散点图
        ax.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Residuals (True - Predicted)')
        ax.set_title(f'{set_name.capitalize()} Set - Residuals')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    residuals_path = os.path.join(save_dir, 'predictions_residuals.pdf')
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.savefig(residuals_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存残差图: {residuals_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 打印统计信息 ==========
    print("\n" + "="*70)
    print("📊 回归预测统计信息")
    print("="*70)
    for set_name, metrics in metrics_summary.items():
        print(f"\n{set_name.upper()} SET:")
        print(f"  样本数: {len(data[set_name])}")
        print(f"  MAE:    {metrics['MAE']:.6f}")
        print(f"  RMSE:   {metrics['RMSE']:.6f}")
        print(f"  R²:     {metrics['R²']:.6f}")
    print("="*70 + "\n")


def plot_classification_predictions(data, save_dir, show):
    """绘制分类任务的预测结果"""

    n_sets = len(data)

    # ========== 图1: 混淆矩阵 ==========
    fig, axes = plt.subplots(1, n_sets, figsize=(6*n_sets, 5))
    if n_sets == 1:
        axes = [axes]

    metrics_summary = {}

    for idx, (set_name, df) in enumerate(data.items()):
        ax = axes[idx]

        y_true = df['target'].values.astype(int)
        y_pred = (df['prediction'].values > 0.5).astype(int)

        # 计算指标
        metrics = calculate_classification_metrics(y_true, df['prediction'].values)
        metrics_summary[set_name] = metrics

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar=True, square=True, annot_kws={"size": 16})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{set_name.capitalize()} Set - Confusion Matrix')

    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'predictions_confusion_matrix.pdf')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.savefig(cm_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存混淆矩阵: {cm_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 图2: 预测概率分布 ==========
    fig, axes = plt.subplots(1, n_sets, figsize=(6*n_sets, 5))
    if n_sets == 1:
        axes = [axes]

    for idx, (set_name, df) in enumerate(data.items()):
        ax = axes[idx]

        y_true = df['target'].values.astype(int)
        y_pred_proba = df['prediction'].values

        # 分别绘制两个类别的概率分布
        class_0_probs = y_pred_proba[y_true == 0]
        class_1_probs = y_pred_proba[y_true == 1]

        ax.hist(class_0_probs, bins=30, alpha=0.6, label='Class 0 (True)', color='blue', edgecolor='black')
        ax.hist(class_1_probs, bins=30, alpha=0.6, label='Class 1 (True)', color='red', edgecolor='black')
        ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold=0.5')

        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{set_name.capitalize()} Set - Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'predictions_probability_distribution.pdf')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.savefig(dist_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存概率分布图: {dist_path}")

    if show:
        plt.show()
    plt.close()

    # ========== 图3: ROC曲线（如果可计算） ==========
    try:
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(8, 8))

        for set_name, df in data.items():
            y_true = df['target'].values.astype(int)
            y_pred_proba = df['prediction'].values

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, linewidth=2,
                   label=f'{set_name.capitalize()} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        roc_path = os.path.join(save_dir, 'predictions_roc_curve.pdf')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.savefig(roc_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f"✅ 保存ROC曲线: {roc_path}")

        if show:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"⚠️  无法绘制ROC曲线: {e}")

    # ========== 打印统计信息 ==========
    print("\n" + "="*70)
    print("📊 分类预测统计信息")
    print("="*70)
    for set_name, metrics in metrics_summary.items():
        print(f"\n{set_name.upper()} SET:")
        print(f"  样本数:   {len(data[set_name])}")
        print(f"  Accuracy:  {metrics['Accuracy']:.6f}")
        print(f"  Precision: {metrics['Precision']:.6f}")
        print(f"  Recall:    {metrics['Recall']:.6f}")
        print(f"  F1 Score:  {metrics['F1']:.6f}")
        if 'AUC-ROC' in metrics:
            print(f"  AUC-ROC:   {metrics['AUC-ROC']:.6f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制预测结果图')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='包含predictions_*.csv文件的输出目录')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存图片的目录（默认与output_dir相同）')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图片（仅保存）')

    args = parser.parse_args()

    print("="*60)
    print("📈 开始绘制预测结果图")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    print(f"保存目录: {args.save_dir or args.output_dir}")
    print()

    plot_regression_predictions(
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        show=not args.no_show
    )

    print("✅ 所有图表生成完成！")
