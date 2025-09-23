import numpy as np
import matplotlib.pyplot as plt

# 设置图形样式
plt.style.use('default')
# plt.rcParams['font.size'] = 12

# 定义数据
x_labels = ['2.7k', '5.5k', '11k', '22k']  # x轴标签
x_positions = np.arange(len(x_labels))  # x轴位置

# 定义数据 - 每个子列表对应一个k值的y值
y_data = [
    [6.52, 6.27, 5.37, 4.98],  # k=2
    [6.10, 6.00, 5.49, 5.06],  # k=3
    [5.99, 5.39, 5.23, 5.19]   # k=4
]

# 定义k值列表和颜色
k_values = [2, 3, 4]
# colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 更现代的颜色
colors = ['#1a237e', '#1565c0', '#29b6f6']  # 深海军蓝 → 明亮蓝 → 天蓝
markers = ['o', 's', '^']  # 不同的标记形状
line_styles = ['-', '--', '-.']  # 不同的线型

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每条曲线
for i, (k, color, marker, line_style) in enumerate(zip(k_values, colors, markers, line_styles)):
    y_values = y_data[i]
    
    # 处理零值数据（用NaN替换0，这样不会绘制该点）
    y_values_processed = [y if y != 0 else np.nan for y in y_values]
    
    # 绘制曲线
    ax.plot(x_positions, y_values_processed, 
            marker=marker, 
            color=color, 
            linestyle=line_style,
            linewidth=2.5,
            markersize=10,
            markerfacecolor='white',
            markeredgewidth=2,
            label=f'k = {k}',
            zorder=3)

# 设置图形属性
ax.set_xlabel('Forget Set Size', fontsize=20)
ax.set_ylabel('Overall Score', fontsize=20)
# 设置x轴
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)

# 设置y轴范围
ax.set_ylim(4.5, 6.9)
ax.set_yticks(np.arange(4.5, 6.9, 0.5))

ax.tick_params(axis='both', which='major', labelsize=20)

# 添加网格
ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

# 添加图例
ax.legend(loc='upper right', 
        frameon=True, 
        fancybox=True, 
        shadow=True,
        prop={'size': 16, 'weight': 'bold'},  # 使用prop字典设置字体属性
        handlelength=4.0,      # 设置图例中线的长度
        handleheight=2.0,      # 设置图例中线的高度
        ncol=1,                # 设置列数（1为单列）
        markerscale=1.5,
    )  

# 移除 spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 添加数据标签
for i, y_values in enumerate(y_data):
    for j, y in enumerate(y_values):
        if y != 0:
            # 设置默认值（大多数点在上方）
            offset = 10
            va = 'bottom'
            
            # 特殊情况：需要放在下方的点
            if (i == 2 and j != len(y_values) - 1):  # 第三条曲线，除了最后一个点
                offset = -10
                va = 'top'
            elif i == 0 and j >= 2:  # 第一条曲线的后两个点（索引2和3）
                offset = -8
                va = 'top'
            
            ax.annotate(f'{y:.2f}', 
                    xy=(j, y), 
                    xytext=(0, offset),
                    textcoords='offset points',
                    ha='center', 
                    va=va,
                    fontsize=15,
                    color=colors[i],
            )
# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig("/data/home/jvnting/cnpo/plot/scal_pii.pdf", dpi=300)