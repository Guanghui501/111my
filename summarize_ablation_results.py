#!/usr/bin/env python
"""
消融实验结果汇总脚本
从各个实验目录中提取结果并生成对比表格
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def extract_best_metrics(exp_dir):
    """从实验目录中提取最佳指标"""
    history_val_file = os.path.join(exp_dir, 'history_val.json')

    if not os.path.exists(history_val_file):
        return None

    with open(history_val_file, 'r') as f:
        history = json.load(f)

    # 检测任务类型
    metrics_available = list(history.keys())
    is_classification = 'accuracy' in metrics_available

    results = {
        'exp_dir': os.path.basename(exp_dir),
    }

    if is_classification:
        # 分类任务
        results['best_val_accuracy'] = max(history['accuracy']) if 'accuracy' in history else None
        results['best_val_precision'] = max(history['precision']) if 'precision' in history else None
        results['best_val_recall'] = max(history['recall']) if 'recall' in history else None
        results['best_val_loss'] = min(history['loss']) if 'loss' in history else None
        results['final_val_accuracy'] = history['accuracy'][-1] if 'accuracy' in history else None
    else:
        # 回归任务
        results['best_val_mae'] = min(history['mae']) if 'mae' in history else None
        results['best_val_loss'] = min(history['loss']) if 'loss' in history else None
        results['final_val_mae'] = history['mae'][-1] if 'mae' in history else None
        results['final_val_loss'] = history['loss'][-1] if 'loss' in history else None

    # 训练轮次
    results['total_epochs'] = len(history['loss']) if 'loss' in history else None

    return results


def extract_test_metrics(exp_dir):
    """从预测文件中提取测试集指标"""
    # 尝试三个版本的预测文件
    pred_files = [
        'predictions_best_val_model_test.csv',
        'predictions_best_test_model_test.csv',
        'prediction_results_test_set.csv'
    ]

    for pred_file in pred_files:
        pred_path = os.path.join(exp_dir, pred_file)
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)

            # 移除列名中的空格
            df.columns = df.columns.str.strip()

            # 检测任务类型
            unique_targets = df['target'].nunique()
            is_classification = unique_targets <= 10 and set(df['target'].unique()).issubset({0, 1})

            if is_classification:
                # 分类任务
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                y_true = df['target'].values.astype(int)
                y_pred = (df['prediction'].values > 0.5).astype(int)

                return {
                    'test_accuracy': accuracy_score(y_true, y_pred),
                    'test_precision': precision_score(y_true, y_pred, zero_division=0),
                    'test_recall': recall_score(y_true, y_pred, zero_division=0),
                    'test_f1': f1_score(y_true, y_pred, zero_division=0),
                }
            else:
                # 回归任务
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                y_true = df['target'].values
                y_pred = df['prediction'].values

                return {
                    'test_mae': mean_absolute_error(y_true, y_pred),
                    'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'test_r2': r2_score(y_true, y_pred),
                }

    return {}


def summarize_ablation_experiments(ablation_dir):
    """汇总所有消融实验结果"""

    print("\n" + "="*80)
    print("📊 消融实验结果汇总")
    print("="*80)

    # 实验配置映射
    exp_configs = {
        'exp1_text_concat_baseline': {
            'name': 'Exp-1: Text Simple Concat (Baseline)',
            'cross_modal': '❌',
            'middle_fusion': '❌',
            'fine_grained': '❌',
        },
        'exp2_late_fusion': {
            'name': 'Exp-2: +Late Fusion',
            'cross_modal': '✅',
            'middle_fusion': '❌',
            'fine_grained': '❌',
        },
        'exp3_middle_fusion': {
            'name': 'Exp-3: +Middle Fusion (创新1)',
            'cross_modal': '✅',
            'middle_fusion': '✅',
            'fine_grained': '❌',
        },
        'exp4_fine_grained': {
            'name': 'Exp-4: +Fine-Grained (创新2)',
            'cross_modal': '✅',
            'middle_fusion': '❌',
            'fine_grained': '✅',
        },
        'exp5_full_model': {
            'name': 'Exp-5: Full Model',
            'cross_modal': '✅',
            'middle_fusion': '✅',
            'fine_grained': '✅',
        },
    }

    # 收集所有实验结果
    all_results = []

    for exp_id, config in exp_configs.items():
        exp_dir = os.path.join(ablation_dir, exp_id)

        if not os.path.exists(exp_dir):
            print(f"⚠️  未找到实验目录: {exp_dir}")
            continue

        print(f"\n处理: {config['name']}...")

        # 提取验证集指标
        val_metrics = extract_best_metrics(exp_dir)
        if val_metrics is None:
            print(f"  ❌ 未找到验证集结果")
            continue

        # 提取测试集指标
        test_metrics = extract_test_metrics(exp_dir)

        # 合并结果
        result = {
            'Experiment': config['name'],
            'Cross-Modal': config['cross_modal'],
            'Middle Fusion': config['middle_fusion'],
            'Fine-Grained': config['fine_grained'],
        }
        result.update(val_metrics)
        result.update(test_metrics)

        all_results.append(result)
        print(f"  ✅ 成功提取结果")

    if not all_results:
        print("\n❌ 未找到任何实验结果")
        return

    # 创建DataFrame
    df = pd.DataFrame(all_results)

    # 保存到CSV
    output_csv = os.path.join(ablation_dir, 'ablation_summary.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 结果已保存到: {output_csv}")

    # 打印汇总表格
    print("\n" + "="*80)
    print("📊 验证集最佳结果对比")
    print("="*80)

    # 选择要显示的列
    if 'best_val_mae' in df.columns:
        # 回归任务
        display_cols = ['Experiment', 'Cross-Modal', 'Middle Fusion', 'Fine-Grained',
                       'best_val_mae', 'test_mae', 'test_r2']
        if all(col in df.columns for col in display_cols):
            print(df[display_cols].to_string(index=False))
    elif 'best_val_accuracy' in df.columns:
        # 分类任务
        display_cols = ['Experiment', 'Cross-Modal', 'Middle Fusion', 'Fine-Grained',
                       'best_val_accuracy', 'test_accuracy', 'test_f1']
        if all(col in df.columns for col in display_cols):
            print(df[display_cols].to_string(index=False))

    # 计算相对提升
    print("\n" + "="*80)
    print("📈 相对性能提升分析")
    print("="*80)

    if 'test_mae' in df.columns and len(df) >= 2:
        baseline_mae = df.iloc[0]['test_mae']
        print(f"\n基线 (Exp-1) MAE: {baseline_mae:.6f}")
        print("\n相对提升:")
        for idx, row in df.iterrows():
            if idx == 0:
                continue
            improvement = (baseline_mae - row['test_mae']) / baseline_mae * 100
            print(f"  {row['Experiment']}: {improvement:+.2f}% (MAE: {row['test_mae']:.6f})")

    elif 'test_accuracy' in df.columns and len(df) >= 2:
        baseline_acc = df.iloc[0]['test_accuracy']
        print(f"\n基线 (Exp-1) Accuracy: {baseline_acc:.4f}")
        print("\n绝对提升:")
        for idx, row in df.iterrows():
            if idx == 0:
                continue
            improvement = (row['test_accuracy'] - baseline_acc) * 100
            print(f"  {row['Experiment']}: {improvement:+.2f}% (Acc: {row['test_accuracy']:.4f})")

    print("\n" + "="*80)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='汇总消融实验结果')
    parser.add_argument('--ablation_dir', type=str, required=True,
                        help='消融实验根目录')

    args = parser.parse_args()

    df = summarize_ablation_experiments(args.ablation_dir)

    if df is not None:
        print(f"\n✅ 汇总完成！共 {len(df)} 个实验")
        print(f"详细结果保存在: {args.ablation_dir}/ablation_summary.csv")
