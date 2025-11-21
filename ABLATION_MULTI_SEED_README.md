# 多种子消融实验脚本使用说明

本目录包含用于运行多种子消融实验的自动化脚本。

## 📁 文件说明

### 1. `run_ablation_multi_seed.sh`
**主启动脚本** - 自动运行所有消融实验

**实验配置:**
- **Exp-1: Baseline** - 文本简单拼接（无跨模态注意力）
- **Exp-2: +Late Fusion** - 添加晚期跨模态注意力
- **Exp-3: +Middle Fusion** - Late + 中期融合（创新1）
- **Exp-4: +Fine-Grained** - Late + 细粒度注意力（创新2）

**随机种子:** 42, 123, 7

**总计:** 4个实验 × 3个种子 = 12个训练任务

### 2. `check_ablation_multi_seed_progress.sh`
**进度监控脚本** - 检查所有实验的运行状态

### 3. `summarize_multi_seed_results.py`
**结果汇总脚本** - 生成包含均值和标准差的CSV报告

---

## 🚀 快速开始

### Step 1: 启动所有实验

```bash
./run_ablation_multi_seed.sh
```

这将：
- 在后台启动12个训练任务
- 每个任务独立运行，互不干扰
- 日志保存到各自的目录
- 输出所有进程的PID

**输出示例:**
```
============================================================================
🚀 启动消融实验（多种子版本）
============================================================================
时间: 2025-11-21 10:30:00
数据集: jarvis/mbj_bandgap
实验配置: 4个实验 × 3个种子 = 12个训练任务
随机种子: 42 123 7
基础输出目录: ./ablation_multi_seed
============================================================================

启动: Exp-1: Baseline (seed=42)
  输出目录: ./ablation_multi_seed/exp1_seed42
  配置: cross_modal=False, middle_fusion=False, fine_grained=False
  后台进程PID: 12345
  日志文件: ./ablation_multi_seed/exp1_seed42/nohup.log
...
```

### Step 2: 监控实验进度

```bash
./check_ablation_multi_seed_progress.sh
```

**输出内容:**
1. 后台进程状态（运行中/已完成）
2. 各实验详细进度（完成轮数、最佳性能）
3. 最新日志摘要（最后5行）
4. 结果汇总表
5. 磁盘使用情况

**实时监控:**
```bash
# 每60秒自动刷新
watch -n 60 ./check_ablation_multi_seed_progress.sh
```

### Step 3: 汇总结果

```bash
python summarize_multi_seed_results.py --ablation_dir ./ablation_multi_seed
```

**生成文件:**
- `ablation_summary.csv` - 简明汇总（均值±标准差）
- `ablation_detailed.csv` - 详细结果（每个种子）

**输出示例:**
```
============================================================================
📊 多种子消融实验结果汇总
============================================================================

Exp-1: Baseline
------------------------------------------------------------
  Seed  42: mae=0.2850 (epoch 78, total 100 epochs)
  Seed 123: mae=0.2835 (epoch 82, total 100 epochs)
  Seed   7: mae=0.2862 (epoch 75, total 100 epochs)

  统计: mae = 0.2849 ± 0.0014
  完成数: 3/3

...

============================================================================
📈 改进效果分析
============================================================================

基线 (Baseline): 0.2849 ± 0.0014

+Late Fusion        : 0.2774 ± 0.0012 → 降低 0.0075 (2.63%)
+Middle Fusion      : 0.2703 ± 0.0015 → 降低 0.0146 (5.13%)
+Fine-Grained       : 0.2688 ± 0.0018 → 降低 0.0161 (5.65%)

🏆 最佳配置: +Fine-Grained (MAE = 0.2688 ± 0.0018)
```

---

## 📊 目录结构

```
ablation_multi_seed/
├── exp1_seed42/
│   ├── nohup.log
│   ├── history_val.json
│   ├── history_train.json
│   ├── best_model.pt
│   └── ...
├── exp1_seed123/
├── exp1_seed7/
├── exp2_seed42/
├── ...
├── exp4_seed7/
├── launch_log_YYYYMMDD_HHMMSS.txt
├── running_pids.txt
├── ablation_summary.csv
└── ablation_detailed.csv
```

---

## 🔧 常用命令

### 查看特定实验的日志
```bash
# 查看 Exp-1, Seed-42
tail -f ./ablation_multi_seed/exp1_seed42/nohup.log

# 查看 Exp-3, Seed-123
tail -f ./ablation_multi_seed/exp3_seed123/nohup.log
```

### 查看所有运行中的进程
```bash
# 读取PID文件
cat ./ablation_multi_seed/running_pids.txt

# 查看进程状态
ps -p $(cat ./ablation_multi_seed/running_pids.txt | tr '\n' ',' | sed 's/,$//') -o pid,stat,etime,cmd
```

### 查看GPU使用情况
```bash
# 实时监控
nvidia-smi

# 每秒刷新
watch -n 1 nvidia-smi
```

### 终止所有实验
```bash
# 读取并终止所有进程
kill $(cat ./ablation_multi_seed/running_pids.txt)

# 强制终止
kill -9 $(cat ./ablation_multi_seed/running_pids.txt)
```

### 终止特定实验
```bash
# 查找特定实验的PID
ps aux | grep "exp1_seed42"

# 终止该进程
kill <PID>
```

---

## 📈 结果CSV格式

### `ablation_summary.csv`
| Experiment | Description | Cross-Modal | Middle Fusion | Fine-Grained | Completed | Best MAE (Mean±Std) |
|------------|-------------|-------------|---------------|--------------|-----------|---------------------|
| Baseline | Text Simple Concat | ✗ | ✗ | ✗ | 3/3 | 0.2849±0.0014 |
| +Late | Late fusion | ✓ | ✗ | ✗ | 3/3 | 0.2774±0.0012 |
| +Middle | Late + Middle fusion | ✓ | ✓ | ✗ | 3/3 | 0.2703±0.0015 |
| +FineGrained | Late + Fine-grained | ✓ | ✗ | ✓ | 3/3 | 0.2688±0.0018 |

### `ablation_detailed.csv`
| Experiment | Seed | Total Epochs | Best Epoch | Best Val MAE | Final Val MAE | Best Train Loss | Final Train Loss |
|------------|------|--------------|------------|--------------|---------------|-----------------|------------------|
| Baseline | 42 | 100 | 78 | 0.2850 | 0.2855 | 0.0234 | 0.0210 |
| Baseline | 123 | 100 | 82 | 0.2835 | 0.2840 | 0.0228 | 0.0205 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## ⚠️ 注意事项

1. **GPU资源**: 12个任务同时运行需要足够的GPU资源
   - 建议: 使用 `CUDA_VISIBLE_DEVICES` 限制每个任务的GPU
   - 或者修改脚本添加串行执行逻辑

2. **磁盘空间**: 每个实验约占用500MB-2GB，总计约12-24GB

3. **训练时间**: 单个实验约需1-3小时，12个任务并行约3-5小时

4. **Early Stopping**: 设置为150轮耐心值，配合100轮epochs

5. **随机种子**: 使用42, 123, 7保证结果可重现

---

## 🐛 故障排除

### 问题1: 脚本无法执行
```bash
# 添加执行权限
chmod +x run_ablation_multi_seed.sh
chmod +x check_ablation_multi_seed_progress.sh
```

### 问题2: Python脚本找不到模块
```bash
# 检查环境
which python
python -c "import torch; import numpy; import pandas; print('OK')"

# 如果缺少模块
pip install torch numpy pandas
```

### 问题3: 所有进程已完成但结果不全
```bash
# 检查各个日志文件是否有错误
for log in ./ablation_multi_seed/exp*/nohup.log; do
    echo "=== $log ==="
    tail -20 "$log" | grep -i "error\|exception\|failed"
done
```

### 问题4: 进程被意外终止
```bash
# 检查系统日志
dmesg | tail -50

# 检查是否OOM
grep -i "out of memory" /var/log/syslog
```

---

## 📧 联系方式

如有问题，请检查：
1. 各实验的 `nohup.log` 文件
2. 主启动日志 `launch_log_*.txt`
3. 训练历史文件 `history_val.json` 和 `history_train.json`
