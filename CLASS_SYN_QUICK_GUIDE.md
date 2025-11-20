# 分类数据集训练指南 - class/syn 格式

## 📋 概述

本指南说明如何创建类似 `jarvis/mbj_bandgap` 的标准分类数据集格式 `class/syn`，并进行训练。

数据集结构：
```
dataset/
└── class/
    └── syn/
        ├── cif/
        │   ├── 0.cif
        │   ├── 1.cif
        │   ├── 2.cif
        │   └── ...
        └── description.csv    (id, target, description)
```

---

## 🚀 完整流程（3步）

### 第一步：生成分类CSV

```bash
# 使用generate_classification_csv.py生成CSV
python generate_classification_csv.py \
    --class1_dir ./positive_samples \
    --class0_dir ./negative_samples \
    --output classification_data.csv \
    --workers 4
```

### 第二步：准备标准数据集格式

```bash
# 方法1: 合并所有CIF文件到一个目录
mkdir all_cifs
cp positive_samples/*.cif all_cifs/
cp negative_samples/*.cif all_cifs/

# 方法2: 使用prepare_classification_dataset.py自动创建
python prepare_classification_dataset.py \
    --csv_file classification_data.csv \
    --cif_dir ./all_cifs \
    --output_dir ./dataset \
    --dataset_name syn
```

**输出结构：**
```
dataset/
└── class/
    └── syn/
        ├── cif/              # CIF文件（以Id命名）
        │   ├── 0.cif
        │   ├── 1.cif
        │   └── ...
        └── description.csv   # 描述文件
```

**description.csv 格式：**
```csv
id,target,description
0,1,"SnO2 is Rutile structured and crystallizes..."
1,0,"MgO is Rock Salt structured and crystallizes..."
2,1,"Fe2O3 is Corundum structured and crystallizes..."
...
```

### 第三步：训练模型

```bash
# 基础训练（仅结构）
python train_with_cross_modal_attention.py \
    --dataset class \
    --property syn \
    --root_dir ./dataset \
    --classification 1 \
    --use_cross_modal False \
    --batch_size 32 \
    --epochs 500 \
    --output_dir ./output_class_syn

# 多模态训练（结构+文本，推荐✨）
python train_with_cross_modal_attention.py \
    --dataset class \
    --property syn \
    --root_dir ./dataset \
    --classification 1 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --early_stopping_patience 50 \
    --output_dir ./output_class_syn_multimodal
```

---

## 📝 详细说明

### 1. prepare_classification_dataset.py 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--csv_file` | 分类CSV文件路径 | `classification_data.csv` |
| `--cif_dir` | 包含所有CIF文件的目录 | `./all_cifs` |
| `--output_dir` | 输出根目录 | `./dataset` |
| `--dataset_name` | 数据集名称 | `syn`, `metal_oxide` |

**作用：**
- 复制CIF文件并重命名为 `{Id}.cif`
- 创建标准格式的 `description.csv`
- 生成配置文件和使用说明

### 2. 训练参数说明

**数据集参数：**
- `--dataset class` - 指定为分类数据集
- `--property syn` - 数据集名称（对应 class/syn）
- `--root_dir ./dataset` - 数据集根目录

**分类任务参数：**
- `--classification 1` - 启用分类模式（必须）
- `--classification_threshold 0.5` - 分类阈值

**模型参数：**
- `--use_cross_modal True` - 使用跨模态注意力（结构+文本）
- `--cross_modal_num_heads 4` - 注意力头数
- `--alignn_layers 4` - ALIGNN层数
- `--gcn_layers 4` - GCN层数

**训练参数：**
- `--batch_size 32` - 批次大小
- `--epochs 500` - 训练轮数
- `--learning_rate 0.001` - 学习率
- `--early_stopping_patience 50` - Early stopping

---

## 🎯 不同配置示例

### 配置1：仅结构信息（基准）

```bash
python train_with_cross_modal_attention.py \
    --dataset class \
    --property syn \
    --root_dir ./dataset \
    --classification 1 \
    --use_cross_modal False \
    --batch_size 64 \
    --epochs 500 \
    --output_dir ./output_structure_only
```

**特点**：
- 只使用图神经网络
- 不使用文本描述
- 适合作为基准对比

### 配置2：结构+文本（晚期融合，推荐）

```bash
python train_with_cross_modal_attention.py \
    --dataset class \
    --property syn \
    --root_dir ./dataset \
    --classification 1 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./output_late_fusion
```

**特点**：
- 使用跨模态注意力
- 结构和文本特征融合
- 性能通常最好

### 配置3：完整融合（所有机制）

```bash
python train_with_cross_modal_attention.py \
    --dataset class \
    --property syn \
    --root_dir ./dataset \
    --classification 1 \
    --use_cross_modal True \
    --cross_modal_num_heads 8 \
    --use_middle_fusion True \
    --middle_fusion_layers "2,3" \
    --use_fine_grained_attention True \
    --fine_grained_num_heads 8 \
    --batch_size 16 \
    --epochs 1000 \
    --learning_rate 0.0005 \
    --output_dir ./output_full_fusion
```

**特点**：
- 晚期融合 + 中期融合 + 细粒度注意力
- 最复杂的模型
- 需要更多训练时间

---

## 📊 创建多个分类数据集

您可以创建多个不同的分类数据集：

```bash
# 数据集1: class/syn（合成 vs 天然）
python prepare_classification_dataset.py \
    --csv_file classification_syn.csv \
    --cif_dir ./all_cifs_syn \
    --output_dir ./dataset \
    --dataset_name syn

# 数据集2: class/metal_oxide（金属氧化物分类）
python prepare_classification_dataset.py \
    --csv_file classification_metal_oxide.csv \
    --cif_dir ./all_cifs_metal \
    --output_dir ./dataset \
    --dataset_name metal_oxide

# 数据集3: class/magnetic（磁性材料分类）
python prepare_classification_dataset.py \
    --csv_file classification_magnetic.csv \
    --cif_dir ./all_cifs_mag \
    --output_dir ./dataset \
    --dataset_name magnetic
```

**目录结构：**
```
dataset/
└── class/
    ├── syn/
    │   ├── cif/
    │   └── description.csv
    ├── metal_oxide/
    │   ├── cif/
    │   └── description.csv
    └── magnetic/
        ├── cif/
        └── description.csv
```

**训练不同数据集：**
```bash
# 训练 syn 数据集
python train_with_cross_modal_attention.py --dataset class --property syn ...

# 训练 metal_oxide 数据集
python train_with_cross_modal_attention.py --dataset class --property metal_oxide ...

# 训练 magnetic 数据集
python train_with_cross_modal_attention.py --dataset class --property magnetic ...
```

---

## 🔍 验证数据集

训练前验证数据集结构：

```bash
# 检查目录结构
ls -lh dataset/class/syn/
ls -lh dataset/class/syn/cif/ | head

# 检查description.csv
head dataset/class/syn/description.csv

# 统计样本数
wc -l dataset/class/syn/description.csv
ls dataset/class/syn/cif/ | wc -l
```

---

## 📈 查看训练结果

训练完成后：

```bash
# 查看输出目录
cd output_class_syn_multimodal

# 查看配置
cat config.json

# 查看预测结果
head test_predictions.csv

# 计算准确率（Python）
python -c "
import pandas as pd
df = pd.read_csv('test_predictions.csv')
y_true = df['target'].values
y_pred = (df['prediction'] > 0.5).astype(int)
accuracy = (y_true == y_pred).mean()
print(f'Test Accuracy: {accuracy:.4f}')
"
```

---

## 🆚 与回归任务的对比

| 方面 | 回归任务 | 分类任务 (class/syn) |
|------|----------|----------------------|
| **数据集格式** | `jarvis/mbj_bandgap` | `class/syn` |
| **目标值** | 连续值（如1.23, 2.45） | 0 或 1 |
| **description.csv** | id, composition, target, description | id, target, description |
| **训练参数** | `--classification 0` | `--classification 1` |
| **损失函数** | MSE | BCE (Binary Cross Entropy) |
| **评估指标** | MAE, RMSE | Accuracy, Precision, Recall |

---

## 💡 最佳实践

1. **数据准备**
   - 确保两个类别样本数大致平衡
   - 检查所有CIF文件有效性
   - 验证文本描述质量

2. **超参数选择**
   - 小数据集（<1000）：batch_size=16-32, 增加weight_decay
   - 大数据集（>5000）：batch_size=64, 降低weight_decay
   - 使用early stopping避免过拟合

3. **实验设置**
   - 先用仅结构模型建立基准
   - 再添加文本模态对比提升
   - 记录所有配置和结果

4. **结果分析**
   - 查看混淆矩阵
   - 分析错误分类样本
   - 可视化注意力权重

---

## 🔗 相关文件

- `generate_classification_csv.py` - 生成分类CSV
- `prepare_classification_dataset.py` - 创建标准数据集格式
- `train_with_cross_modal_attention.py` - 训练脚本
- `CLASSIFICATION_TRAINING_GUIDE.txt` - 详细训练指南
- `QUICKSTART_CLASSIFICATION.md` - 快速开始指南

---

**现在您可以像使用 jarvis/mbj_bandgap 一样使用 class/syn 进行分类训练了！** 🎉
