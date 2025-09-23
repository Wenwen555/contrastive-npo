import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hmean

sns.set_style("whitegrid")

categories = ['Privacy', 'Fluency', 'Coherence']
# values = [5.61, 6.97, 6.54]
retrain_values = [8.62, 6.12, 5.19]
pretrain_values = [1.37, 7.77, 7.44]

# 计算角度（三角雷达图是 3 个点，120° 间隔）
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles = [angle + np.pi/2 for angle in angles]  # 旋转 90° 使第一个点朝上
angles += angles[:1]  # 闭合图形

# 画图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

# 绘制坐标轴（从中心到三个顶点）
for angle in angles[:-1]:  # 跳过闭合点
    ax.plot([angle, angle], [0, 10], color='black', linewidth=1.3, alpha=0.8)
    # 添加箭头（使用极坐标方式）
    ax.annotate('', 
                xy=(angle, 10.5),  # 箭头终点（极坐标）
                xytext=(angle, 10),  # 箭头起点（极坐标）
                arrowprops=dict(arrowstyle="->", color='black', linewidth=1),
                annotation_clip=False)

# 绘制等距参考线和刻度标签
levels = [2, 4, 6, 8, 10]
for level in levels:
    ax.plot(angles, [level]*4, color='gray', alpha=0.3, linewidth=0.5)
    # 在三个轴上添加刻度标签（极坐标方式）
    for angle in angles[:-1]:
        ax.text(angle, level, str(level),  # 直接使用极坐标定位
                ha='center', va='center', 
                fontsize=8, color='black', alpha=0.7,
                transform=ax.transData)

# 绘制数据三角形
ax.fill(angles, values + values[:1], color="skyblue", alpha=0.3, label='Data')
ax.plot(angles, values + values[:1], color="skyblue", alpha=1, linewidth=1)

# 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_yticks([])  # 隐藏默认数值刻度
ax.set_ylim(0, 10.5)  # 为箭头留出空间

# 关闭极坐标的圆形网格线和边框
ax.grid(False)
ax.spines['polar'].set_visible(False)

# 计算调和平均数并标注在中心
hmean_val = hmean(values)
print(hmean_val)
# ax.text(0, 0, f'hmean = {hmean_val:.2f}',
#         ha='center', va='center', fontsize=12, 
#         bbox=dict(facecolor='white', alpha=0.8),
#         transform=ax.transData)  # 中心文本也使用数据坐标

# plt.savefig('triple/simnpo-checkpoint98.svg', dpi=900, bbox_inches='tight')
# plt.show()