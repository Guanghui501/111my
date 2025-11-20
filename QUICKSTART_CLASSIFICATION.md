# 快速开始：晶体材料分类训练

## 📋 完整流程

从CIF文件到训练完成的完整步骤指南

---

## 第一步：生成分类CSV（已完成✅）

```bash
# 使用多进程加速处理
python generate_classification_csv.py \
    --class1_dir ./class1_cifs \
    --class0_dir ./class0_cifs \
    --output classification_data.csv \
    --workers 4
```

生成的`classification_data.csv`包含：
- **Id**: 0, 1, 2, 3, ...
- **Composition**: SnO2, Fe2O3, ...
- **prop**: 1 (类别1) 或 0 (类别0)
- **Description**: 结构描述文本
- **File_Name**: 原始CIF文件名

---

## 第二步：准备CIF文件

将所有CIF文件合并到一个目录：

```bash
# 创建合并目录
mkdir all_cifs

# 复制所有CIF文件
cp class1_cifs/*.cif all_cifs/
cp class0_cifs/*.cif all_cifs/

# 验证文件数量
ls all_cifs/*.cif | wc -l
```

---

## 第三步：转换CSV为JSON

```bash
python convert_csv_to_json.py \
    --csv_file classification_data.csv \
    --cif_dir ./all_cifs \
    --output_json classification_data.json
```

输出的`classification_data.json`格式：
```json
[
  {
    "jid": "0",
    "composition": "SnO2",
    "atoms": {...},
    "target": 1.0,
    "description": "SnO2 is Rutile structured...",
    "file_name": "SnO2.cif"
  },
  ...
]
```

---

## 第四步：组织数据集目录

创建标准的数据集目录结构：

```bash
# 创建数据集目录
mkdir -p dataset/classification

# 移动JSON文件
mv classification_data.json dataset/classification/

# 数据集目录结构
dataset/
└── classification/
    └── classification_data.json
```

---

## 第五步：开始训练 🚀

### 5.1 基础分类训练（仅结构信息）

```bash
python train_with_cross_modal_attention.py \
    --dataset user_data \
    --property classification \
    --root_dir ./dataset/classification \
    --classification 1 \
    --use_cross_modal False \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.001 \
    --output_dir ./output_structure_only
```

### 5.2 多模态训练（结构+文本，推荐✨）

```bash
python train_with_cross_modal_attention.py \
    --dataset user_data \
    --property classification \
    --root_dir ./dataset/classification \
    --classification 1 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --early_stopping_patience 50 \
    --output_dir ./output_multimodal
```

### 5.3 完整配置（所有融合机制）

```bash
python train_with_cross_modal_attention.py \
    --dataset user_data \
    --property classification \
    --root_dir ./dataset/classification \
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
    --weight_decay 1e-4 \
    --early_stopping_patience 100 \
    --output_dir ./output_full_fusion
```

---

## 第六步：查看训练结果

训练完成后，检查输出目录：

```bash
cd output_multimodal

# 查看配置
cat config.json

# 查看测试集预测
head test_predictions.csv
```

输出文件：
- `config.json` - 训练配置
- `checkpoint_best.pt` - 最佳模型
- `train_predictions.csv` - 训练集预测
- `val_predictions.csv` - 验证集预测
- `test_predictions.csv` - 测试集预测

---

## 📊 评估模型性能

使用Python分析结果：

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 读取测试集预测
df = pd.read_csv('output_multimodal/test_predictions.csv')

# 提取真实标签和预测结果
y_true = df['target'].values
y_pred = (df['prediction'] > 0.5).astype(int)  # 阈值0.5

# 计算指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("\n混淆矩阵:")
print(cm)
```

---

## 🎯 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--classification` | 启用分类模式 | 1 |
| `--classification_threshold` | 分类阈值 | 0.5 |
| `--use_cross_modal` | 使用文本信息 | True |
| `--cross_modal_num_heads` | 注意力头数 | 4-8 |
| `--batch_size` | 批次大小 | 16-64 |
| `--learning_rate` | 学习率 | 0.001 |
| `--epochs` | 训练轮数 | 500-1000 |
| `--early_stopping_patience` | Early stopping | 50-100 |

---

## 🔧 常见问题

### Q1: 如何处理类别不平衡？

如果类别1有100个样本，类别0有400个样本：

**方法1**: 调整分类阈值
```bash
--classification_threshold 0.7  # 提高阈值
```

**方法2**: 平衡数据集
```python
# 在生成CSV时确保两个类别样本数接近
```

### Q2: 训练速度太慢？

**优化建议**:
1. 增加batch_size: `--batch_size 64`
2. 减少注意力头数: `--cross_modal_num_heads 2`
3. 关闭高级特性: `--use_middle_fusion False`
4. 使用GPU（如果可用）

### Q3: 准确率一直在50%？

**可能原因**:
1. 学习率过大 → 降低到 0.0005
2. 模型未收敛 → 增加训练轮数
3. 数据质量问题 → 检查描述文本和CIF文件

### Q4: 如何对比不同方法？

```bash
# 方法1: 仅结构
python train_with... --use_cross_modal False --output_dir output_structure

# 方法2: 结构+文本
python train_with... --use_cross_modal True --output_dir output_multimodal

# 比较结果
python compare_results.py output_structure output_multimodal
```

---

## 📈 性能基准

基于JARVIS数据集的典型性能：

| 方法 | 准确率 | 训练时间 |
|------|--------|----------|
| 仅结构 | 75-80% | 2-3小时 |
| 结构+文本（晚期融合） | 82-88% | 3-4小时 |
| 完整融合 | 85-92% | 5-6小时 |

*注：实际性能取决于数据集质量和分类任务难度*

---

## 🎓 下一步

1. **超参数调优**: 尝试不同的学习率、batch_size等
2. **消融实验**: 对比有无文本描述的效果
3. **可解释性分析**: 查看注意力权重
4. **迁移学习**: 在其他分类任务上微调

---

## 📚 相关文档

- `CLASSIFICATION_CSV_README.txt` - CSV生成详细说明
- `CLASSIFICATION_TRAINING_GUIDE.txt` - 完整训练指南
- `INSTALL_GUIDE.txt` - 依赖安装指南

---

## 💡 提示

- 首次训练建议使用小数据集（<500样本）测试
- 使用`--early_stopping_patience`避免过拟合
- 保存好每次实验的配置和结果
- GPU可显著加速训练（3-5倍）

---

**祝训练顺利！🚀**
