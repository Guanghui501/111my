========================================================================
分类任务CSV生成脚本使用说明
========================================================================

脚本名称: generate_classification_csv.py
功能: 使用robocrystallographer-0.2.12为晶体结构生成文本描述，
      并创建用于分类任务的CSV文件

========================================================================
CSV文件格式
========================================================================

CSV文件包含以下列:
  - Id: 结构的唯一标识符 (格式: {label}_{index})
  - Composition: 化学组成 (例如: SnO2, Fe2O3)
  - prop: 分类标签 (1 或 0)
  - Description: robocrystallographer生成的结构描述文本
  - File_Name: 原始CIF文件名

========================================================================
使用方法
========================================================================

1. 准备数据:
   创建两个文件夹，分别存放不同类别的CIF文件:

   class1_cifs/          # 标签为1的结构
     ├── structure1.cif
     ├── structure2.cif
     └── ...

   class0_cifs/          # 标签为0的结构
     ├── structure1.cif
     ├── structure2.cif
     └── ...

2. 运行脚本:

   基本用法（单进程）:
   python generate_classification_csv.py \
       --class1_dir ./class1_cifs \
       --class0_dir ./class0_cifs \
       --output classification_data.csv

   加速处理（使用多进程，推荐）:
   python generate_classification_csv.py \
       --class1_dir ./class1_cifs \
       --class0_dir ./class0_cifs \
       --output classification_data.csv \
       --workers 4

   自动使用所有CPU核心（最快）:
   python generate_classification_csv.py \
       --class1_dir ./class1_cifs \
       --class0_dir ./class0_cifs \
       --output classification_data.csv \
       --workers -1

3. 查看帮助:
   python generate_classification_csv.py --help

4. 查看示例:
   python generate_classification_csv.py --test

========================================================================
示例命令
========================================================================

# 示例1: 基本用法（单进程）
python generate_classification_csv.py \
    --class1_dir /path/to/positive_samples \
    --class0_dir /path/to/negative_samples \
    --output my_classification_data.csv

# 示例2: 使用4个进程加速（推荐，处理大量文件时）
python generate_classification_csv.py \
    --class1_dir ./metal_oxides \
    --class0_dir ./non_metal_oxides \
    --output oxide_classification.csv \
    --workers 4

# 示例3: 自动使用所有CPU核心（最快）
python generate_classification_csv.py \
    --class1_dir ./class1_cifs \
    --class0_dir ./class0_cifs \
    --output data.csv \
    --workers -1

# 示例4: 查看系统CPU核心数
python generate_classification_csv.py --test

========================================================================
输出示例
========================================================================

CSV文件内容示例:

Id,Composition,prop,Description,File_Name
1_0,SnO2,1,"SnO2 is Rutile structured and crystallizes in the tetragonal P4_2/mnm space group. The structure is three-dimensional...",SnO2.cif
1_1,Fe2O3,1,"Fe2O3 is Corundum structured and crystallizes in the trigonal R-3c space group...",Fe2O3.cif
0_0,SiO2,0,"SiO2 is Quartz structured and crystallizes in the trigonal P3_121 space group...",SiO2.cif

========================================================================
注意事项
========================================================================

1. CIF文件要求:
   - 文件必须是有效的CIF格式
   - 文件扩展名必须是 .cif
   - 确保CIF文件包含完整的晶体结构信息

2. 依赖项:
   脚本使用以下Python包:
   - pymatgen (晶体结构处理)
   - robocrystallographer (描述生成)
   - pandas (CSV处理)
   - tqdm (进度条显示)

3. 性能优化:
   - 单进程模式: 每个结构约需1-6秒
   - 多进程模式: 可显著加速处理（推荐使用）
   - 建议根据CPU核心数设置 --workers 参数
   - 使用 --workers -1 自动使用所有CPU核心

   性能对比示例（100个文件）:
   - 单进程 (--workers 1): 约500秒
   - 4进程 (--workers 4): 约125秒 (4倍加速)
   - 8进程 (--workers 8): 约62秒 (8倍加速)

4. 错误处理:
   - 如果某个CIF文件无法处理，脚本会跳过该文件并继续
   - 多进程模式下错误信息会被抑制以保持输出清晰

========================================================================
输出信息
========================================================================

脚本运行时会显示:
1. 正在处理的目录和文件数量
2. 处理进度条
3. 成功处理的文件数量
4. CSV文件保存位置
5. 数据统计信息
6. 数据预览(前几行)
7. 分类标签分布

========================================================================
技术细节
========================================================================

描述生成设置:
- describe_mineral: True (描述矿物信息)
- describe_component_makeup: True (描述组分构成)
- describe_components: True (描述组分详情)
- describe_symmetry_labels: False (简化输出)
- describe_oxidation_states: True (包含氧化态)
- describe_bond_lengths: True (包含键长信息)

========================================================================
