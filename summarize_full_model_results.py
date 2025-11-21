#!/usr/bin/env python
"""
全模块训练结果汇总脚本
生成包含均值和标准差的详细报告
"""

import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_model_results(base_dir, seed):
    """加载单个Full Model的结果"""
    model_dir = Path(base_dir) / f"full_model_seed{seed}"

    if not model_dir.exists():
        return None

    history_val_file = model_dir / "history_val.json"
    history_train_file = model_dir / "history_train.json"

    if not history_val_file.exists():
        return None

    try:
        with open(history_val_file, 'r') as f:
            val_history = json.load(f)

        with open(history_train_file, 'r') as f:
            train_history = json.load(f)

        # 检测任务类型
        if 'mae' in val_history:
            task_type = 'regression'
            metric_name = 'mae'
            val_metrics = val_history['mae']
            best_val = min(val_metrics)
            best_epoch = val_metrics.index(best_val)
        elif 'accuracy' in val_history:
            task_type = 'classification'
            metric_name = 'accuracy'
            val_metrics = val_history['accuracy']
            best_val = max(val_metrics)
            best_epoch = val_metrics.index(best_val)
        else:
            return None

        # 提取关键指标
        result = {
            'task_type': task_type,
            'metric_name': metric_name,
            'total_epochs': len(val_history['epochs']),
            'best_epoch': val_history['epochs'][best_epoch],
            'best_val': best_val,
            'final_val': val_metrics[-1],
            'best_train_loss': train_history['loss'][best_epoch],
            'final_train_loss': train_history['loss'][-1],
        }

        # 添加额外指标（如果存在）
        if task_type == 'regression':
            if 'rmse' in val_history:
                result['best_val_rmse'] = min(val_history['rmse'])
                result['final_val_rmse'] = val_history['rmse'][-1]
        elif task_type == 'classification':
            if 'precision' in val_history:
                result['best_val_precision'] = max(val_history['precision'])
                result['final_val_precision'] = val_history['precision'][-1]
            if 'recall' in val_history:
                result['best_val_recall'] = max(val_history['recall'])
                result['final_val_recall'] = val_history['recall'][-1]
            if 'f1' in val_history:
                result['best_val_f1'] = max(val_history['f1'])
                result['final_val_f1'] = val_history['f1'][-1]

        return result

    except Exception as e:
        print(f"警告: 读取 {model_dir} 时出错: {e}")
        return None


def summarize_results(base_dir):
    """汇总所有Full Model结果"""

    base_dir = Path(base_dir)

    seeds = [42, 123, 7]

    # 收集所有结果
    all_results = []
    task_type = None
    metric_name = None

    print("="*80)
    print("📊 Full Model训练结果汇总")
    print("="*80)
    print(f"\n基础目录: {base_dir}\n")

    print("Full Model (所有模块启用)")
    print("-" * 60)

    for seed in seeds:
        result = load_model_results(base_dir, seed)

        if result is not None:
            if task_type is None:
                task_type = result['task_type']
                metric_name = result['metric_name']

            all_results.append(result)

            # 打印单个种子的结果
            print(f"  Seed {seed:3d}: "
                  f"{metric_name}={result['best_val']:.4f} "
                  f"(epoch {result['best_epoch']}, "
                  f"total {result['total_epochs']} epochs)")
        else:
            print(f"  Seed {seed:3d}: 未完成或数据缺失")

    if not all_results:
        print("\n❌ 没有可用的结果数据！")
        return

    # 计算统计量
    best_vals = [r['best_val'] for r in all_results]
    mean_val = np.mean(best_vals)
    std_val = np.std(best_vals, ddof=1) if len(best_vals) > 1 else 0

    print(f"\n  统计: {metric_name} = {mean_val:.4f} ± {std_val:.4f}")
    print(f"  完成数: {len(all_results)}/{len(seeds)}")

    # ========================================================================
    # 生成CSV报告
    # ========================================================================
    print("\n" + "="*80)
    print("📄 生成CSV报告")
    print("="*80)

    # CSV 1: 简明汇总
    summary_data = {
        'Model': 'Full Model',
        'Description': 'All modules enabled (Late + Middle + Fine-Grained)',
        'Completed': f"{len(all_results)}/3",
        f'Best {metric_name.upper()} (Mean±Std)': f"{mean_val:.4f}±{std_val:.4f}",
        'Min': f"{min(best_vals):.4f}",
        'Max': f"{max(best_vals):.4f}",
    }

    df_summary = pd.DataFrame([summary_data])

    summary_csv = base_dir / "full_model_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n✅ 简明汇总已保存: {summary_csv}")

    # CSV 2: 详细结果
    detailed_rows = []
    for i, seed in enumerate(seeds):
        if i < len(all_results):
            result = all_results[i]
            row = {
                'Seed': seed,
                'Total Epochs': result['total_epochs'],
                'Best Epoch': result['best_epoch'],
                f'Best Val {metric_name.upper()}': result['best_val'],
                f'Final Val {metric_name.upper()}': result['final_val'],
                'Best Train Loss': result['best_train_loss'],
                'Final Train Loss': result['final_train_loss'],
            }

            # 添加额外指标
            if task_type == 'regression' and 'best_val_rmse' in result:
                row['Best Val RMSE'] = result['best_val_rmse']
                row['Final Val RMSE'] = result['final_val_rmse']
            elif task_type == 'classification':
                if 'best_val_precision' in result:
                    row['Best Val Precision'] = result['best_val_precision']
                if 'best_val_recall' in result:
                    row['Best Val Recall'] = result['best_val_recall']
                if 'best_val_f1' in result:
                    row['Best Val F1'] = result['best_val_f1']

            detailed_rows.append(row)

    df_detailed = pd.DataFrame(detailed_rows)

    detailed_csv = base_dir / "full_model_detailed.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"✅ 详细结果已保存: {detailed_csv}")

    # ========================================================================
    # 打印详细表格
    # ========================================================================
    print("\n" + "="*80)
    print("📊 详细结果表")
    print("="*80)
    print()

    print(df_detailed.to_string(index=False))

    # ========================================================================
    # 统计信息
    # ========================================================================
    print("\n" + "="*80)
    print("📈 统计信息")
    print("="*80)
    print()

    print(f"模型配置: Full Model (所有模块)")
    print(f"  - Late Fusion: ✓")
    print(f"  - Middle Fusion: ✓")
    print(f"  - Fine-Grained Attention: ✓")
    print()

    print(f"训练任务: {len(all_results)}/{len(seeds)} 完成")
    print()

    print(f"性能指标 ({metric_name.upper()}):")
    print(f"  - 平均: {mean_val:.4f}")
    print(f"  - 标准差: {std_val:.4f}")
    print(f"  - 最小值: {min(best_vals):.4f} (Seed {seeds[best_vals.index(min(best_vals))]})")
    print(f"  - 最大值: {max(best_vals):.4f} (Seed {seeds[best_vals.index(max(best_vals))]})")
    print()

    # 计算平均训练轮数
    avg_epochs = np.mean([r['total_epochs'] for r in all_results])
    print(f"平均训练轮数: {avg_epochs:.1f}")

    print("\n" + "="*80)
    print("✅ 汇总完成！")
    print("="*80)
    print()


def main():
    parser = argparse.ArgumentParser(description='Full Model训练结果汇总')
    parser.add_argument('--model_dir', type=str, default='./full_model_multi_seed',
                        help='Full Model训练基础目录')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"错误: 目录不存在: {args.model_dir}")
        sys.exit(1)

    summarize_results(args.model_dir)


if __name__ == '__main__':
    main()
