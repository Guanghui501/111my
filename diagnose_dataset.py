#!/usr/bin/env python
"""
诊断分类数据集 - 检查 class/syn 格式是否正确

使用方法:
  python diagnose_dataset.py --root_dir ./dataset --dataset class --property syn
"""

import os
import csv
import argparse
from pathlib import Path


def diagnose_dataset(root_dir, dataset, property_name):
    """诊断数据集结构和内容"""

    print("=" * 70)
    print(f"诊断数据集: {dataset}/{property_name}")
    print("=" * 70)

    # 构建路径
    if dataset.lower() == 'class':
        cif_dir = os.path.join(root_dir, f'class/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'class/{property_name}/description.csv')
    elif dataset.lower() == 'jarvis':
        cif_dir = os.path.join(root_dir, f'jarvis/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'jarvis/{property_name}/description.csv')
    else:
        print(f"不支持的数据集类型: {dataset}")
        return

    print(f"\n1. 检查路径:")
    print(f"   CIF目录: {cif_dir}")
    print(f"   描述文件: {id_prop_file}")

    # 检查目录是否存在
    print(f"\n2. 检查目录存在性:")
    if os.path.exists(cif_dir):
        print(f"   ✓ CIF目录存在")
    else:
        print(f"   ✗ CIF目录不存在: {cif_dir}")
        return

    if os.path.exists(id_prop_file):
        print(f"   ✓ 描述文件存在")
    else:
        print(f"   ✗ 描述文件不存在: {id_prop_file}")
        return

    # 读取CSV文件
    print(f"\n3. 检查CSV文件内容:")
    with open(id_prop_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"   CSV列名: {headings}")
    print(f"   总行数: {len(data)}")

    # 检查CSV格式
    print(f"\n4. 检查CSV格式:")
    format_errors = []
    for i, row in enumerate(data[:10]):  # 检查前10行
        if len(row) < 3:
            format_errors.append(f"   行 {i}: 列数不足 ({len(row)} 列)")
        elif len(row) > 3:
            print(f"   警告 - 行 {i}: 列数过多 ({len(row)} 列)，将使用前3列")

    if format_errors:
        print("   ✗ 发现格式错误:")
        for err in format_errors:
            print(err)
    else:
        print(f"   ✓ 前10行格式正确 (3列: id, target, description)")

    # 显示示例数据
    print(f"\n5. 数据示例 (前5行):")
    for i, row in enumerate(data[:5]):
        if len(row) >= 3:
            print(f"   行 {i}: id={row[0]}, target={row[1]}, desc={row[2][:50]}...")
        else:
            print(f"   行 {i}: {row}")

    # 检查CIF文件
    print(f"\n6. 检查CIF文件:")
    cif_files = list(Path(cif_dir).glob("*.cif"))
    print(f"   CIF文件总数: {len(cif_files)}")

    # 检查CIF文件是否与CSV匹配
    print(f"\n7. 检查CIF文件与CSV的匹配:")
    missing_cifs = []
    existing_cifs = []

    for i, row in enumerate(data[:20]):  # 检查前20个
        if len(row) >= 1:
            file_id = row[0]
            cif_path = os.path.join(cif_dir, f'{file_id}.cif')
            if os.path.exists(cif_path):
                existing_cifs.append(file_id)
            else:
                missing_cifs.append(file_id)

    print(f"   检查前20个样本:")
    print(f"     ✓ 找到CIF文件: {len(existing_cifs)}")
    print(f"     ✗ 缺失CIF文件: {len(missing_cifs)}")

    if missing_cifs:
        print(f"   缺失的CIF文件ID (前5个): {missing_cifs[:5]}")

    # 检查标签分布
    print(f"\n8. 检查标签分布:")
    targets = []
    for row in data:
        if len(row) >= 2:
            try:
                targets.append(float(row[1]))
            except:
                pass

    if targets:
        from collections import Counter
        target_counts = Counter(targets)
        print(f"   标签分布:")
        for label, count in sorted(target_counts.items()):
            print(f"     标签 {int(label)}: {count} 个样本")

    # 总结
    print("\n" + "=" * 70)
    print("诊断总结:")
    print("=" * 70)

    total_samples = len(data)
    matching_samples = 0

    for row in data:
        if len(row) >= 1:
            file_id = row[0]
            cif_path = os.path.join(cif_dir, f'{file_id}.cif')
            if os.path.exists(cif_path) and len(row) >= 3:
                matching_samples += 1

    print(f"CSV中的样本总数: {total_samples}")
    print(f"完整有效的样本: {matching_samples}")
    print(f"预计可用样本: {matching_samples} ({matching_samples/total_samples*100:.1f}%)")

    if matching_samples < total_samples:
        print(f"\n⚠ 注意: 有 {total_samples - matching_samples} 个样本可能无法加载")
        print(f"  原因可能是:")
        print(f"    1. CIF文件不存在或路径不对")
        print(f"    2. CSV格式错误（列数不对）")
        print(f"    3. 数据损坏")
    else:
        print(f"\n✓ 数据集看起来完整，可以开始训练！")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='诊断分类数据集')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='class',
                        help='数据集类型 (class, jarvis)')
    parser.add_argument('--property', type=str, default='syn',
                        help='数据集名称 (syn, mbj_bandgap等)')

    args = parser.parse_args()

    diagnose_dataset(args.root_dir, args.dataset, args.property)


if __name__ == "__main__":
    main()
