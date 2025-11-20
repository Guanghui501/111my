#!/usr/bin/env python
"""
脚本功能：生成用于分类任务的描述文本CSV文件
使用robocrystallographer-0.2.12生成晶体结构的文本描述

CSV格式：Id, Composition, prop, Description, File_Name
- Id: 结构的唯一标识符
- Composition: 化学组成
- prop: 分类标签（1或0）
- Description: robocrystallographer生成的文本描述
- File_Name: CIF文件名

使用方法：
python generate_classification_csv.py --class1_dir /path/to/class1_cifs --class0_dir /path/to/class0_cifs --output classification_data.csv

加速处理（使用多进程）：
python generate_classification_csv.py --class1_dir /path/to/class1_cifs --class0_dir /path/to/class0_cifs --output classification_data.csv --workers 4
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加robocrystallographer到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robocrystallographer-0.2.12'))

from pymatgen.core.structure import Structure
from robocrys import StructureCondenser, StructureDescriber

# 忽略警告
warnings.filterwarnings('ignore')


def get_structure_description(cif_file_path):
    """
    从CIF文件生成晶体结构描述

    参数:
        cif_file_path: CIF文件的路径

    返回:
        tuple: (composition, description) 或 (None, None) 如果失败
    """
    try:
        # 从CIF文件加载结构
        structure = Structure.from_file(cif_file_path)

        # 获取化学组成
        composition = structure.composition.reduced_formula

        # 尝试添加氧化态
        try:
            if not any(hasattr(s, "oxi_state") for s in structure.composition.elements):
                structure.add_oxidation_state_by_guess(max_sites=-80)
        except:
            pass  # 如果无法添加氧化态，继续处理

        # 使用robocrystallographer生成描述
        condenser = StructureCondenser()
        describer = StructureDescriber(
            describe_mineral=True,
            describe_component_makeup=True,
            describe_components=True,
            describe_symmetry_labels=False,  # 简化描述
            describe_oxidation_states=True,
            describe_bond_lengths=True
        )

        condensed_structure = condenser.condense_structure(structure)
        description = describer.describe(condensed_structure)

        # 移除描述中的换行符，替换为空格
        description = description.replace('\n', ' ').strip()

        return composition, description

    except Exception as e:
        # 在多进程中，避免打印太多错误信息
        return None, None


def process_single_file(args):
    """
    处理单个CIF文件的包装函数（用于多进程）

    参数:
        args: tuple (cif_file, label)

    返回:
        dict 或 None
    """
    cif_file, label = args
    composition, description = get_structure_description(str(cif_file))

    if composition and description:
        return {
            'Composition': composition,
            'prop': label,
            'Description': description,
            'File_Name': cif_file.name
        }
    return None


def process_cif_directory(directory_path, label, workers=1):
    """
    处理目录中的所有CIF文件

    参数:
        directory_path: 包含CIF文件的目录路径
        label: 分类标签（1或0）
        workers: 并行处理的进程数（默认为1，即单进程）

    返回:
        list: 包含字典的列表，每个字典代表一个结构
    """
    data_list = []
    cif_files = list(Path(directory_path).glob("*.cif"))

    if len(cif_files) == 0:
        print(f"警告: 在 {directory_path} 中没有找到CIF文件")
        return data_list

    print(f"\n处理 {directory_path} 中的 {len(cif_files)} 个CIF文件 (标签={label})...")

    if workers > 1:
        print(f"使用 {workers} 个进程并行处理...")

        # 准备参数列表
        args_list = [(cif_file, label) for cif_file in cif_files]

        # 使用多进程处理
        with Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(cif_files),
                desc=f"处理标签{label}的文件"
            ))

        # 过滤掉None结果
        data_list = [r for r in results if r is not None]
    else:
        # 单进程处理（原有逻辑）
        for cif_file in tqdm(cif_files, desc=f"处理标签{label}的文件"):
            composition, description = get_structure_description(str(cif_file))

            if composition and description:
                data_entry = {
                    'Composition': composition,
                    'prop': label,
                    'Description': description,
                    'File_Name': cif_file.name
                }
                data_list.append(data_entry)

    print(f"成功处理 {len(data_list)}/{len(cif_files)} 个文件")
    return data_list


def generate_classification_csv(class1_dir, class0_dir, output_file, workers=1):
    """
    生成分类任务的CSV文件

    参数:
        class1_dir: 标签为1的CIF文件目录
        class0_dir: 标签为0的CIF文件目录
        output_file: 输出CSV文件路径
        workers: 并行处理的进程数
    """
    print("=" * 60)
    print("开始生成分类任务CSV文件")
    if workers > 1:
        print(f"多进程模式: 使用 {workers} 个进程")
    print("=" * 60)

    # 检查目录是否存在
    if not os.path.exists(class1_dir):
        raise FileNotFoundError(f"目录不存在: {class1_dir}")
    if not os.path.exists(class0_dir):
        raise FileNotFoundError(f"目录不存在: {class0_dir}")

    # 处理两个目录中的CIF文件
    data_class1 = process_cif_directory(class1_dir, label=1, workers=workers)
    data_class0 = process_cif_directory(class0_dir, label=0, workers=workers)

    # 合并数据
    all_data = data_class1 + data_class0

    if len(all_data) == 0:
        print("错误: 没有成功处理任何文件")
        return

    # 创建DataFrame
    df = pd.DataFrame(all_data)

    # 添加从0开始的全局Id
    df.insert(0, 'Id', range(len(df)))

    # 确保列的顺序
    df = df[['Id', 'Composition', 'prop', 'Description', 'File_Name']]

    # 保存到CSV
    df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print(f"CSV文件已生成: {output_file}")
    print(f"总样本数: {len(df)}")
    print(f"  - 标签=1: {len(data_class1)} 个样本")
    print(f"  - 标签=0: {len(data_class0)} 个样本")
    print("=" * 60)

    # 显示前几行数据
    print("\n数据预览:")
    print(df.head())

    # 显示统计信息
    print("\n分类标签分布:")
    print(df['prop'].value_counts())


def main():
    parser = argparse.ArgumentParser(
        description='生成用于分类任务的晶体结构描述CSV文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（单进程）
  python generate_classification_csv.py --class1_dir ./positive_samples --class0_dir ./negative_samples --output classification_data.csv

  # 使用4个进程加速处理
  python generate_classification_csv.py --class1_dir ./positive_samples --class0_dir ./negative_samples --output classification_data.csv --workers 4

  # 自动使用所有CPU核心
  python generate_classification_csv.py --class1_dir ./positive_samples --class0_dir ./negative_samples --output classification_data.csv --workers -1

  # 使用测试数据（如果您还没有准备CIF文件）
  python generate_classification_csv.py --test
        """
    )

    parser.add_argument('--class1_dir', type=str,
                        help='包含标签为1的CIF文件的目录路径')
    parser.add_argument('--class0_dir', type=str,
                        help='包含标签为0的CIF文件的目录路径')
    parser.add_argument('--output', type=str, default='classification_data.csv',
                        help='输出CSV文件路径 (默认: classification_data.csv)')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行处理的进程数 (默认: 1, -1表示使用所有CPU核心)')
    parser.add_argument('--test', action='store_true',
                        help='使用测试模式创建示例目录结构')

    args = parser.parse_args()

    if args.test:
        print("测试模式: 请先准备两个包含CIF文件的目录")
        print("\n目录结构示例:")
        print("  class1_cifs/")
        print("    ├── structure1.cif")
        print("    ├── structure2.cif")
        print("    └── ...")
        print("  class0_cifs/")
        print("    ├── structure1.cif")
        print("    ├── structure2.cif")
        print("    └── ...")
        print("\n基本用法:")
        print("  python generate_classification_csv.py --class1_dir ./class1_cifs --class0_dir ./class0_cifs --output my_data.csv")
        print("\n加速处理（推荐）:")
        print("  python generate_classification_csv.py --class1_dir ./class1_cifs --class0_dir ./class0_cifs --output my_data.csv --workers 4")
        print(f"\n您的系统有 {cpu_count()} 个CPU核心")
        sys.exit(0)

    if not args.class1_dir or not args.class0_dir:
        parser.error("需要指定 --class1_dir 和 --class0_dir 参数，或使用 --test 查看示例")

    # 处理workers参数
    workers = args.workers
    if workers == -1:
        workers = cpu_count()
        print(f"自动检测到 {workers} 个CPU核心")
    elif workers < 1:
        workers = 1
    elif workers > cpu_count():
        print(f"警告: 指定的进程数 {workers} 超过CPU核心数 {cpu_count()}，将使用 {cpu_count()} 个进程")
        workers = cpu_count()

    # 生成CSV文件
    generate_classification_csv(args.class1_dir, args.class0_dir, args.output, workers=workers)


if __name__ == "__main__":
    main()
