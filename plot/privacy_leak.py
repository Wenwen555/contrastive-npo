import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 导入 patches 模块
import json
import numpy as np

def extract_min_k_data(log):
    """将eval()返回的log字典转换为Min-K%的DataFrame"""
    min_k_columns = [f"Min-{k}%" for k in [5, 10, 20, 30, 40, 50, 60]]
    
    dfs = []
    for split in ['forget', 'retain', 'holdout']:
        split_df = pd.DataFrame(log[split])
        # 提取所有Min-K%列和文本
        split_df = split_df[['text'] + min_k_columns].copy()
        split_df['dataset'] = split
        dfs.append(split_df)
    return pd.concat(dfs)


def plot_min_k_distribution(df, k_percent=5, figsize=(10,6)):
    """绘制指定K%的概率密度分布图"""
    plt.figure(figsize=figsize)
    metric = f'Min-{k_percent}%'
    
    # 检查数据有效性
    if 'dataset' not in df.columns:
        raise ValueError("DataFrame中缺少'dataset'列")
    
    valid_datasets = ['forget', 'retain', 'holdout']
    if not set(df['dataset'].unique()).issuperset(set(valid_datasets)):
        raise ValueError(f"'dataset'列应包含 {valid_datasets} 中的值")
    
    # 过滤有效数据
    plot_df = df[df['dataset'].isin(valid_datasets)].copy()
    
    import ipdb
    ipdb.set_trace()
    # 绘制密度图
    ax = sns.kdeplot(
        data=plot_df, 
        x=metric, 
        hue='dataset',
        palette={'forget': 'red', 'retain': 'blue', 'holdout': 'green'},
        common_norm=False,
        fill=True,
        alpha=0.2,
        linewidth=1.5,
        hue_order=valid_datasets  # 确保顺序一致
    )
    
    # 美化图形
    plt.title(f'Min-{k_percent}% Probability Distribution Comparison', pad=20)
    plt.xlabel(f'Min-{k_percent}% Probability (-log scale)')
    # plt.ylabel('Density')
    # plt.grid(True, alpha=0.3)
    sns.despine()
    
    # 安全处理图例
    if ax.get_legend_handles_labels()[0]:
        # 重映射图例标签
        legend_map = {
            'forget': 'Forget Set', 
            'retain': 'Retain Set', 
            'holdout': 'Holdout Set'
        }
        ax.legend(
            title='Dataset',
            labels=[legend_map.get(l, l) for l in ax.get_legend_handles_labels()[1]]
        )
    else:
        # 手动创建图例作为后备方案
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Forget Set'),
            Line2D([0], [0], color='blue', lw=2, label='Retain Set'),
            Line2D([0], [0], color='green', lw=2, label='Holdout Set')
        ]
        ax.legend(handles=legend_elements, title='Dataset')
    
    return ax

def plot_combined_improved_min_k(df, k_percent=5, figsize=(12,6), bin_width=0.1):
    """绘制叠加柱状图与密度曲线的混合图表"""
    plt.figure(figsize=figsize)
    metric = f'Min-{k_percent}%'
    
    # 数据验证
    valid_datasets = ['forget', 'retain', 'holdout']
    if not set(df['dataset'].unique()).issuperset(set(valid_datasets)):
        raise ValueError(f"数据集必须包含 {valid_datasets}")
    
    # 创建双轴系统
    ax1 = plt.gca()  # 柱状图主坐标轴
    ax2 = ax1.twinx()  # 密度曲线次坐标轴
    
    # --- 第一部分：绘制堆叠柱状图 ---
    # 动态计算分箱边界
    min_val = df[metric].min()
    max_val = df[metric].max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 存储柱状图的patches用于图例
    hist_patches = []
    
    # 按数据集分组绘制
    for i, dataset in enumerate(valid_datasets):
        subset = df[df['dataset'] == dataset]
        color = {'forget': '#FFA540', 'retain': '#4B5ED7', 'holdout': '#37DB79'}[dataset]
        
        # 绘制透明柱体
        n, bins, patches = ax1.hist(
            subset[metric],
            bins=bins,
            color=color,
            alpha=0.2,  # 降低透明度避免遮挡曲线
            edgecolor=color,
            linewidth=0.5,
            density=False,  # 显示实际计数
            stacked=False,
            zorder=2  # 确保柱状图在曲线下方
        )
        hist_patches.append(patches[0])  # 保存第一个patch用于图例
    
    # --- 第二部分：绘制密度曲线 ---
    # 存储密度曲线的lines用于图例
    kde_lines = []
    
    for dataset in valid_datasets:
        subset = df[df['dataset'] == dataset]
        line_color = {'forget': '#FF8700', 'retain': '#071672', 'holdout': '#00B74A'}[dataset]
        
        # 手动绘制每个密度曲线以获取line对象
        kde = sns.kdeplot(
            data=subset,
            x=metric,
            color=line_color,
            linewidth=2,
            alpha=0.8,
            ax=ax2,
            zorder=3,  # 确保曲线在上层
            legend=False  # 禁用自动图例
        )
        kde_lines.append(kde.lines[0])
    
    # --- 图形美化 ---
    # 坐标轴标签
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']  # 或 'Liberation Serif', 'Georgia'
    ax1.set_xlabel(
        f'Min-{k_percent}% Probability (-log scale)',
        fontdict={
            'fontsize': 22,
            'fontweight': 'bold',
        }
    )
    # 隐藏右侧坐标轴
    ax2.set_ylabel('')  # 清空标签
    ax2.set_yticks([])  # 移除刻度
    # 网格设置
    ax1.grid(True, linestyle='--', alpha=0.3, which='both')  # Both major and minor
    ax1.tick_params(axis='both', which='both', length=0)  # Hide tick marks
    ax1.set_xticklabels([])  # Hide x-axis labels
    ax1.set_yticklabels([])  # Hide y-axis labels
    # --- 自定义图例 ---
    # 创建组合图例项（同时显示柱状图和密度曲线的样式）
    legend_handles = []
    for patch in hist_patches:
        # 创建长条形图例句柄
        legend_handle = mpatches.Rectangle(
            (0, 0),  # 起始点坐标（不影响显示）
            width=4.0,  # 长条长度
            height=1.2,  # 长条高度
            facecolor=patch.get_facecolor(),  # 填充颜色
            edgecolor=patch.get_edgecolor(),  # 边框颜色
            alpha=0.6,  # 透明度
            linewidth=1  # 边框宽度
        )
        legend_handles.append(legend_handle)

    # 应用图例
    ax1.legend(
        handles=legend_handles,
        labels=[fr'$\mathbf{{{d.capitalize()}}}$' for d in valid_datasets], 
        loc='upper left',
        frameon=True,
        framealpha=0.8,
        fontsize=18,
        handlelength=6,  # 增大此值使图例长条更长（默认2.0）
        handleheight=2,  # 增大此值使图例长条更粗（默认0.8）
        borderpad=1.2
    )
    
    plt.tight_layout()
    return ax1, ax2

def plot_combined_min_k(df, k_percent=5, figsize=(12,6), bin_width=0.1):
    """绘制叠加柱状图与密度曲线的混合图表"""
    plt.figure(figsize=figsize)
    metric = f'Min-{k_percent}%'
    
    # 数据验证
    valid_datasets = ['forget', 'retain', 'holdout']
    if not set(df['dataset'].unique()).issuperset(set(valid_datasets)):
        raise ValueError(f"数据集必须包含 {valid_datasets}")
    
    # 创建双轴系统
    ax1 = plt.gca()  # 柱状图主坐标轴
    ax2 = ax1.twinx()  # 密度曲线次坐标轴
    
    # --- 第一部分：绘制堆叠柱状图 ---
    # 动态计算分箱边界
    min_val = df[metric].min()
    max_val = df[metric].max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 按数据集分组绘制
    for i, dataset in enumerate(valid_datasets):
        subset = df[df['dataset'] == dataset]
        color = {'forget': '#FFA540', 'retain': '#4B5ED7', 'holdout': '#37DB79'}[dataset]
        
        # 绘制透明柱体
        ax1.hist(
            subset[metric],
            bins=bins,
            color=color,
            alpha=0.2,  # 降低透明度避免遮挡曲线
            edgecolor=color,
            linewidth=0.5,
            # label=f'{dataset.capitalize()} Set (Count)',
            density=False,  # 显示实际计数
            stacked=False,
            zorder=2  # 确保柱状图在曲线下方
        )
    
    # --- 第二部分：绘制密度曲线 ---
    sns.kdeplot(
        data=df,
        x=metric,
        hue='dataset',
        palette={'forget': '#FF8700', 'retain': '#071672', 'holdout': '#00B74A'},
        common_norm=False,
        linewidth=2,
        alpha=0.8,
        ax=ax2,  # 指定到次坐标轴
        hue_order=valid_datasets,
        zorder=3  # 确保曲线在上层
    )
    
    # --- 图形美化 ---
    # 坐标轴标签
    ax1.set_xlabel(f'Min-{k_percent}% Probability (-log scale)')
        # --- 隐藏右侧坐标轴 ---
    ax2.set_ylabel('')  # 清空标签
    ax2.set_yticks([])  # 移除刻度
    # ax1.set_ylabel('Sample Count', color='black')
    # ax2.set_ylabel('Density', color='black')
    # 标题与网格
    # plt.title(f'Combined Distribution of Min-{k_percent}% Probability', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.3, which='both')  # Both major and minor
    ax1.tick_params(axis='both', which='both', length=0)  # Hide tick marks
    ax1.set_xticklabels([])  # Hide x-axis labels
    ax1.set_yticklabels([])  # Hide y-axis labels
    # ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return ax1, ax2

if __name__ == "__main__":
    with open('/data/home/jvnting/cnpo/temp/news/final/cnpo-beta-0.1/4n/checkpoint-1221/privleak/log.json', 'rb') as f:
        log1 = json.load(f)
    with open('/data/home/jvnting/cnpo/temp/news/final/retrained_model/retrained_model/retrained_model/privleak/log.json', 'rb') as f:
        log2 = json.load(f)
    min_k_df = extract_min_k_data(log1)
    # min_k_df_2 = extract_min_k_data(log2)     
    min_k_df = extract_min_k_data(log1)
    plot_combined_improved_min_k(min_k_df, k_percent=40)
    plt.savefig('min_k_improved_distribution_cnpo-gdr-4n-beta=0.1-checkpoint=1221.svg', dpi=900, bbox_inches='tight')
    plot_combined_improved_min_k(min_k_df, k_percent=40)
    plt.savefig('min_k_improved_distribution_cnpo-gdr-4n-beta=0.1-checkpoint=1221.svg', dpi=900, bbox_inches='tight')
    # plot_combined_improved_min_k(min_k_df_2, k_percent=40)
    # plt.savefig('min_k_improved_distribution_retrained.svg', dpi=900, bbox_inches='tight')
    plt.show()