#!/usr/bin/env python
"""
可视化OneCycleLR调度器在不同epoch设置下的学习率变化
用于理解为什么epochs=100比epochs=1000效果更好
"""

import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn as nn
import seaborn as sns

sns.set_style("white")

# 模拟参数
batch_size = 128
num_samples = 1000  # 假设训练集大小
steps_per_epoch = num_samples // batch_size

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

def get_lr_schedule(epochs, warmup_steps=2000):
    """获取给定epoch数下的学习率变化"""
    model = DummyModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)

    total_steps = epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.3,  # 前30%的时间用于上升
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    lr_history = []
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            lr_history.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

    return lr_history

# 生成不同配置的学习率历史
print("生成学习率调度曲线...")
lr_100 = get_lr_schedule(100)
lr_1000 = get_lr_schedule(1000)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 图1: 完整的1000 epoch学习率变化
ax1 = axes[0, 0]
steps_1000 = np.arange(len(lr_1000))
epochs_1000 = steps_1000 / steps_per_epoch
ax1.plot(epochs_1000, lr_1000, 'b-', linewidth=2, label='LR schedule (1000 epochs)')
ax1.axvline(x=453, color='r', linestyle='--', linewidth=2, label='实际停止位置 (epoch 453)')
ax1.axvline(x=348, color='g', linestyle='--', linewidth=1.5, alpha=0.7,
            label='最佳验证点估计 (epoch ~348)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Learning Rate', fontsize=12)
ax1.set_title('配置1: epochs=1000 的学习率调度\n(在epoch 453停止，但调度器设计为1000 epochs)',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 图2: 100 epoch学习率变化
ax2 = axes[0, 1]
steps_100 = np.arange(len(lr_100))
epochs_100 = steps_100 / steps_per_epoch
ax2.plot(epochs_100, lr_100, 'g-', linewidth=2, label='LR schedule (100 epochs)')
ax2.axvline(x=100, color='r', linestyle='--', linewidth=2, label='训练结束位置 (epoch 100)')
ax2.axvline(x=99, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
            label='最佳验证点 (epoch ~99)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Learning Rate', fontsize=12)
ax2.set_title('配置2: epochs=100 的学习率调度\n(完整完成调度周期)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 图3: 对比前500个epoch的学习率
ax3 = axes[1, 0]
epochs_1000_subset = np.arange(0, min(500*steps_per_epoch, len(lr_1000))) / steps_per_epoch
epochs_100_scaled = np.arange(len(lr_100)) / steps_per_epoch
ax3.plot(epochs_1000_subset, lr_1000[:len(epochs_1000_subset)], 'b-', linewidth=2,
         label='epochs=1000 (前500轮)', alpha=0.7)
ax3.plot(epochs_100_scaled, lr_100, 'g-', linewidth=2,
         label='epochs=100 (全部)', alpha=0.7)
ax3.axvline(x=100, color='g', linestyle='--', linewidth=1.5,
            label='100 epoch配置结束')
ax3.axvline(x=453, color='b', linestyle='--', linewidth=1.5,
            label='1000 epoch配置early stop')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('前500轮学习率对比', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 500])

# 图4: 关键区域放大（0-150 epochs）
ax4 = axes[1, 1]
epochs_1000_zoom = np.arange(0, min(150*steps_per_epoch, len(lr_1000))) / steps_per_epoch
ax4.plot(epochs_1000_zoom, lr_1000[:len(epochs_1000_zoom)], 'b-', linewidth=2,
         label='epochs=1000', alpha=0.7)
ax4.plot(epochs_100_scaled, lr_100, 'g-', linewidth=2,
         label='epochs=100', alpha=0.7)
ax4.axvline(x=100, color='g', linestyle='--', linewidth=1.5, alpha=0.5,
            label='100 epoch配置结束')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Learning Rate', fontsize=12)
ax4.set_title('前150轮学习率对比（关键训练期）', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 150])

plt.tight_layout()
plt.savefig('scheduler_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('scheduler_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存: scheduler_comparison.pdf 和 scheduler_comparison.png")

# 打印关键统计信息
print("\n" + "="*60)
print("学习率调度分析")
print("="*60)

print(f"\n配置1 (epochs=1000):")
print(f"  - 调度器设计总步数: {len(lr_1000)} steps = {len(lr_1000)/steps_per_epoch:.0f} epochs")
print(f"  - 实际训练停止: epoch 453")
print(f"  - 停止时学习率: {lr_1000[453*steps_per_epoch-1]:.6f}")
print(f"  - 最佳验证点(约epoch 348)学习率: {lr_1000[348*steps_per_epoch-1]:.6f}")
print(f"  - 调度完成度: {453/1000*100:.1f}%")
print(f"  - 问题: 在调度器中期停止，学习率还在较高水平")

print(f"\n配置2 (epochs=100):")
print(f"  - 调度器设计总步数: {len(lr_100)} steps = {len(lr_100)/steps_per_epoch:.0f} epochs")
print(f"  - 实际训练完成: epoch 100")
print(f"  - 结束时学习率: {lr_100[-1]:.6f}")
print(f"  - 最佳验证点(约epoch 99)学习率: {lr_100[99*steps_per_epoch-1]:.6f}")
print(f"  - 调度完成度: {100/100*100:.1f}%")
print(f"  - 优势: 完整完成调度周期，学习率充分衰减")

print(f"\n学习率对比 (epoch 100时):")
print(f"  - epochs=1000配置: {lr_1000[100*steps_per_epoch-1]:.6f}")
print(f"  - epochs=100配置:  {lr_100[100*steps_per_epoch-1]:.6f}")
print(f"  - 比率: {lr_1000[100*steps_per_epoch-1]/lr_100[100*steps_per_epoch-1]:.2f}x")

print("\n" + "="*60)
print("结论:")
print("="*60)
print("""
1. OneCycleLR调度器根据total_steps来规划学习率变化周期
2. epochs=1000时，调度器计划用1000个epoch完成学习率周期
   - 前300个epoch: 学习率上升
   - 中间400个epoch: 学习率高位运行
   - 后300个epoch: 学习率下降

3. epochs=100时，调度器计划用100个epoch完成学习率周期
   - 前30个epoch: 学习率上升
   - 中间40个epoch: 学习率高位运行
   - 后30个epoch: 学习率下降

4. 关键问题:
   - epochs=1000的配置在epoch 453停止，此时学习率还在中等水平
   - epochs=100的配置完整完成了调度周期，学习率充分衰减
   - 更低的最终学习率帮助模型更好地收敛，减少过拟合

5. 建议:
   - 如果使用early stopping，应该将epochs设置得更接近预期的实际训练轮数
   - 或者使用其他调度器如ReduceLROnPlateau，它会根据验证性能动态调整
   - 当前情况下，epochs=100的配置更合理
""")

plt.show()
