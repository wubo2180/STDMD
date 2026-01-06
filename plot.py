import matplotlib.pyplot as plt
import numpy as np

# Set global plot configurations
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'Times New Roman'
})

# Variants and metrics - 消融实验数据
variants = ["STDMD", "S-STDMD", "T-STDMD"]
colors = ["#E5D897", "#FCEFD7", "#F9F5E2"]  # 蓝色、紫红色、橙色
hatches = ['', '//', '..']  # 添加不同的填充图案

# 消融实验数据 - 三个数据集，三个指标
ablation_data = {
    "WikiMaths": {
        "MSE": [0.765, 0.8021, 0.7893],
        "RMSE": [0.8747, 0.8956, 0.8885],
        "MAPE": [8.50, 9.01, 8.89],
    },
    "EnglandCovid": {
        "MSE": [0.5411, 0.6190, 0.6542],
        "RMSE": [0.7354, 0.7868, 0.8088],
        "MAPE": [8.32, 9.11, 9.45],
    },
    "PedalMe": {
        "MSE": [1.0106, 1.0912, 1.1234],
        "RMSE": [1.0053, 1.0445, 1.0590],
        "MAPE": [13.45, 14.32, 14.87],
    }
}

# Create 1x4 subplots (3个数据集 + 1个综合对比)
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

datasets = list(ablation_data.keys())
dataset_names = ["WikiMaths", "EnglandCovid", "PedalMe"]

# Plot individual datasets (前3个子图)
for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    # 准备数据
    metrics = ["MSE", "RMSE", "MAPE"]
    x = np.arange(len(metrics))
    width = 0.25
    
    # 绘制每个变体的柱状图
    for j, variant in enumerate(variants):
        values = [ablation_data[dataset][metric][j] for metric in metrics]
        bars = ax.bar(x + j * width - width, values, width, 
                     label=variant, color=colors[j], hatch=hatches[j],
                     edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_title(f"{dataset_names[i]}", fontsize=22, pad=15)
    ax.set_xlabel("Metrics", fontsize=20)
    ax.set_ylabel("Values", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)

# 第4个子图：标准化后的综合性能对比
ax = axes[3]

# 计算标准化后的性能指标
standardized_scores = {}
metrics = ["MSE", "RMSE", "MAPE"]

# 首先计算每个指标在所有数据集和变体中的最小值和最大值
metric_ranges = {}
for metric in metrics:
    all_values = []
    for dataset in datasets:
        all_values.extend(ablation_data[dataset][metric])
    metric_ranges[metric] = {
        'min': min(all_values),
        'max': max(all_values)
    }

# 为每个变体计算标准化分数
for variant_idx, variant in enumerate(variants):
    normalized_scores = []
    
    for dataset in datasets:
        for metric in metrics:
            value = ablation_data[dataset][metric][variant_idx]
            min_val = metric_ranges[metric]['min']
            max_val = metric_ranges[metric]['max']
            
            # 标准化到0-1区间，对于MSE/RMSE/MAPE，值越小越好，所以用1减去标准化值
            normalized_score = 1 - (value - min_val) / (max_val - min_val)
            normalized_scores.append(normalized_score)
    
    # 计算平均标准化分数
    standardized_scores[variant] = np.mean(normalized_scores)

# 绘制标准化性能对比
x = np.arange(len(variants))
values = list(standardized_scores.values())
bars = ax.bar(x, values, color=colors, hatch=hatches, 
             edgecolor='black', linewidth=1, alpha=0.8, width=0.6)

ax.set_title("Normalized Performance", fontsize=22, pad=15)
ax.set_xlabel("Model Variants", fontsize=20)
ax.set_ylabel("Normalized Score", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=16)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.tick_params(axis='both', labelsize=16)
ax.set_ylim(0, 1)  # 标准化分数范围0-1

# 在柱状图上标注数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# 添加性能等级参考线
ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, linewidth=1)

# 添加参考线标签
ax.text(len(variants)-0.5, 0.82, 'Excellent', fontsize=12, color='green', alpha=0.7)
ax.text(len(variants)-0.5, 0.62, 'Good', fontsize=12, color='orange', alpha=0.7)
ax.text(len(variants)-0.5, 0.42, 'Fair', fontsize=12, color='red', alpha=0.7)

# 添加图例
fig.legend(variants, loc="upper center", ncol=3, fontsize=20, 
          bbox_to_anchor=(0.5, 1.0), frameon=True, fancybox=True, shadow=True)

# 添加整体标题
# fig.suptitle("Ablation Study: Component-wise Analysis of STDMD Framework", 
#             fontsize=24, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.90])
output_path = "figure/Ablation_Study_STDMD_1x4.pdf"
plt.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
plt.show()