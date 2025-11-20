#!/usr/bin/env python
"""
创建标准分类数据集结构

将分类CSV和CIF文件组织成类似 jarvis/mbj_bandgap 的标准格式：

class/syn/
├── cif/
│   ├── 0.cif
│   ├── 1.cif
│   ├── 2.cif
│   └── ...
└── description.csv

description.csv 格式：
  id, target, description

使用方法：
  python prepare_classification_dataset.py \
      --csv_file classification_data.csv \
      --cif_dir ./all_cifs \
      --output_dir ./dataset \
      --dataset_name syn
"""

import os
import sys
import argparse
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def prepare_classification_dataset(csv_file, cif_dir, output_dir, dataset_name='syn'):
    """
    准备标准格式的分类数据集

    参数:
        csv_file: 分类CSV文件路径（来自generate_classification_csv.py）
        cif_dir: 包含所有CIF文件的目录
        output_dir: 输出根目录
        dataset_name: 数据集名称（默认'syn'）
    """
    print("=" * 70)
    print("创建标准分类数据集结构")
    print("=" * 70)

    # 读取CSV文件
    print(f"\n1. 读取分类CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"   找到 {len(df)} 个样本")
    print(f"   标签分布:")
    print(df['prop'].value_counts().sort_index())

    # 创建输出目录结构
    dataset_path = os.path.join(output_dir, 'class', dataset_name)
    cif_output_dir = os.path.join(dataset_path, 'cif')

    print(f"\n2. 创建目录结构: {dataset_path}")
    os.makedirs(cif_output_dir, exist_ok=True)
    print(f"   ✓ {dataset_path}/")
    print(f"   ✓ {dataset_path}/cif/")

    # 复制CIF文件并重命名为Id.cif
    print(f"\n3. 复制并重命名CIF文件...")
    skipped = 0
    copied = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="复制CIF文件"):
        file_id = row['Id']
        file_name = row['File_Name']

        # 源CIF文件路径
        src_cif = os.path.join(cif_dir, file_name)

        if not os.path.exists(src_cif):
            print(f"   警告: 找不到 {src_cif}")
            skipped += 1
            continue

        # 目标CIF文件路径：使用Id作为文件名
        dst_cif = os.path.join(cif_output_dir, f"{file_id}.cif")

        # 复制文件
        shutil.copy2(src_cif, dst_cif)
        copied += 1

    print(f"   ✓ 成功复制 {copied} 个CIF文件")
    if skipped > 0:
        print(f"   ⚠ 跳过 {skipped} 个文件")

    # 创建description.csv
    print(f"\n4. 创建 description.csv...")
    description_df = df[['Id', 'prop', 'Description']].copy()
    description_df.columns = ['id', 'target', 'description']

    description_path = os.path.join(dataset_path, 'description.csv')
    description_df.to_csv(description_path, index=False)
    print(f"   ✓ 保存到: {description_path}")

    # 显示数据集信息
    print("\n" + "=" * 70)
    print("数据集创建完成！")
    print("=" * 70)
    print(f"\n数据集路径: {dataset_path}")
    print(f"├── cif/          ({copied} 个CIF文件)")
    print(f"└── description.csv")

    print(f"\n数据集统计:")
    print(f"  总样本数: {len(description_df)}")
    print(f"  标签分布:")
    for label, count in description_df['target'].value_counts().sort_index().items():
        print(f"    类别 {int(label)}: {count} 个样本")

    # 显示使用方法
    print("\n" + "=" * 70)
    print("使用方法:")
    print("=" * 70)
    print(f"""
训练命令示例:

python train_with_cross_modal_attention.py \\
    --dataset class \\
    --property {dataset_name} \\
    --root_dir {output_dir} \\
    --classification 1 \\
    --use_cross_modal True \\
    --batch_size 32 \\
    --epochs 500 \\
    --output_dir ./output_class_{dataset_name}

或者使用配置文件:

python train_classification.py \\
    --config config_class_{dataset_name}.json
    """)

    # 创建示例配置文件
    config_file = f"config_class_{dataset_name}.json"
    print(f"\n5. 创建示例配置文件: {config_file}")

    import json
    config = {
        "dataset": "class",
        "property": dataset_name,
        "root_dir": output_dir,
        "classification": 1,
        "classification_threshold": 0.5,
        "batch_size": 32,
        "epochs": 500,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "use_cross_modal": True,
        "cross_modal_num_heads": 4,
        "alignn_layers": 4,
        "gcn_layers": 4,
        "hidden_features": 256,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 123,
        "output_dir": f"./output_class_{dataset_name}",
        "early_stopping_patience": 50
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   ✓ 配置已保存到: {config_file}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='创建标准格式的分类数据集（类似jarvis/mbj_bandgap结构）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
示例:

  # 基本用法
  python prepare_classification_dataset.py \\
      --csv_file classification_data.csv \\
      --cif_dir ./all_cifs \\
      --output_dir ./dataset \\
      --dataset_name syn

  # 结果将创建：
  # dataset/class/syn/cif/        (所有CIF文件，以Id命名)
  # dataset/class/syn/description.csv

  # 然后可以像使用jarvis数据集一样训练：
  python train_with_cross_modal_attention.py \\
      --dataset class \\
      --property syn \\
      --root_dir ./dataset \\
      --classification 1
        """
    )

    parser.add_argument('--csv_file', type=str, required=True,
                        help='分类CSV文件（来自generate_classification_csv.py）')
    parser.add_argument('--cif_dir', type=str, required=True,
                        help='包含所有CIF文件的目录')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='输出根目录')
    parser.add_argument('--dataset_name', type=str, default='syn',
                        help='数据集名称（将创建 class/{dataset_name}）')

    args = parser.parse_args()

    # 验证输入
    if not os.path.exists(args.csv_file):
        print(f"错误: CSV文件不存在: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.cif_dir):
        print(f"错误: CIF目录不存在: {args.cif_dir}")
        sys.exit(1)

    # 执行
    prepare_classification_dataset(
        csv_file=args.csv_file,
        cif_dir=args.cif_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )


if __name__ == "__main__":
    main()
