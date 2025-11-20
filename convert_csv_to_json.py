#!/usr/bin/env python
"""
CSV转JSON转换工具 - 用于分类训练

将generate_classification_csv.py生成的CSV文件转换为训练脚本需要的JSON格式

输入: classification_data.csv
输出: classification_data.json

CSV格式: Id, Composition, prop, Description, File_Name
JSON格式: 列表，每个元素包含 jid, atoms, target, description
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 添加robocrystallographer到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robocrystallographer-0.2.12'))

from pymatgen.core.structure import Structure
from jarvis.core.atoms import Atoms as JarvisAtoms


def cif_to_jarvis_atoms(cif_path):
    """
    从CIF文件转换为JARVIS Atoms对象

    参数:
        cif_path: CIF文件路径

    返回:
        dict: JARVIS Atoms字典表示
    """
    try:
        # 使用pymatgen加载结构
        structure = Structure.from_file(cif_path)

        # 转换为JARVIS Atoms对象
        # 获取晶格矩阵和原子坐标
        lattice_mat = structure.lattice.matrix.tolist()
        elements = [str(site.specie) for site in structure.sites]
        coords = [site.frac_coords.tolist() for site in structure.sites]

        # 创建JARVIS Atoms对象
        jarvis_atoms = JarvisAtoms(
            lattice_mat=lattice_mat,
            elements=elements,
            coords=coords,
            cartesian=False
        )

        return jarvis_atoms.to_dict()

    except Exception as e:
        print(f"处理CIF文件 {cif_path} 时出错: {str(e)}")
        return None


def convert_csv_to_json(csv_file, cif_dir, output_json, verbose=True):
    """
    将分类CSV转换为JSON格式

    参数:
        csv_file: 输入CSV文件路径
        cif_dir: CIF文件目录（包含所有class1和class0的CIF文件）
        output_json: 输出JSON文件路径
        verbose: 是否显示详细信息
    """
    print("=" * 60)
    print("CSV到JSON转换工具")
    print("=" * 60)

    # 读取CSV文件
    if verbose:
        print(f"\n正在读取CSV文件: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"找到 {len(df)} 个样本")

    # 检查CIF目录
    if not os.path.exists(cif_dir):
        raise FileNotFoundError(f"CIF目录不存在: {cif_dir}")

    # 转换数据
    json_data = []
    skipped = 0

    if verbose:
        print(f"\n正在处理CIF文件并转换格式...")
        iterator = tqdm(df.iterrows(), total=len(df), desc="转换进度")
    else:
        iterator = df.iterrows()

    for idx, row in iterator:
        # 查找对应的CIF文件
        cif_filename = row['File_Name']
        cif_path = os.path.join(cif_dir, cif_filename)

        if not os.path.exists(cif_path):
            if verbose:
                print(f"警告: 找不到CIF文件 {cif_path}，跳过")
            skipped += 1
            continue

        # 转换CIF为JARVIS Atoms格式
        atoms_dict = cif_to_jarvis_atoms(cif_path)

        if atoms_dict is None:
            skipped += 1
            continue

        # 构建JSON条目
        json_entry = {
            'jid': str(row['Id']),  # 使用Id作为唯一标识符
            'composition': row['Composition'],
            'atoms': atoms_dict,
            'target': float(row['prop']),  # 分类标签（0或1）
            'description': row['Description'],
            'file_name': cif_filename
        }

        json_data.append(json_entry)

    # 保存JSON文件
    if verbose:
        print(f"\n正在保存JSON文件: {output_json}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"转换完成!")
    print(f"  成功转换: {len(json_data)} 个样本")
    print(f"  跳过: {skipped} 个样本")
    print(f"  输出文件: {output_json}")
    print("=" * 60)

    # 显示数据统计
    if len(json_data) > 0:
        df_converted = pd.DataFrame([
            {'jid': entry['jid'], 'target': entry['target']}
            for entry in json_data
        ])
        print(f"\n分类标签分布:")
        print(df_converted['target'].value_counts().sort_index())
        print(f"\n数据已准备好用于训练!")


def main():
    parser = argparse.ArgumentParser(
        description='将分类CSV文件转换为训练所需的JSON格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python convert_csv_to_json.py \\
      --csv_file classification_data.csv \\
      --cif_dir ./all_cifs \\
      --output_json classification_data.json

  # 如果CIF文件分散在两个目录中，需要先合并
  mkdir all_cifs
  cp class1_cifs/*.cif all_cifs/
  cp class0_cifs/*.cif all_cifs/
  python convert_csv_to_json.py \\
      --csv_file classification_data.csv \\
      --cif_dir ./all_cifs \\
      --output_json classification_data.json
        """
    )

    parser.add_argument('--csv_file', type=str, required=True,
                        help='输入的分类CSV文件路径')
    parser.add_argument('--cif_dir', type=str, required=True,
                        help='包含所有CIF文件的目录')
    parser.add_argument('--output_json', type=str, default='classification_data.json',
                        help='输出JSON文件路径')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，减少输出')

    args = parser.parse_args()

    # 执行转换
    convert_csv_to_json(
        csv_file=args.csv_file,
        cif_dir=args.cif_dir,
        output_json=args.output_json,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
