import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data definitions
models = ['Mistral-7B', 'Gemma-7B', 'Llama-3-8B']
methods = ['SFT', 'ORL w/ SFT', 'ORL w/o SFT']

data = {
    'SFT': {
        'without_h': [38.2, 38.1, 41.9],
        'with_h': [45.1, 40.5, 50.2]
    },
    'ORL w/ SFT': {
        'without_h': [39.2, 38.2, 40.1],
        'with_h': [65.1, 58.5, 68.3]
    },
    'ORL w/o SFT': {
        'without_h': [30.4, 28.6, 35.1],
        'with_h': [44.6, 39.1, 47.3]
    }
}

def create_plot():
    # plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(6, 3.2))  # 减小高度
    
    x = np.arange(len(models)) * 0.28
    width = 0.04
    
    colors = {
        'SFT': '#8884d8',
        'ORL w/ SFT': '#82ca9d',
        'ORL w/o SFT': '#ffc658'
    }
    
    for i, method in enumerate(methods):
        pos_without = x + width * (i*2 - 2.5)
        pos_with = x + width * (i*2 - 1.5)
        
        bar1 = ax.bar(pos_without, data[method]['without_h'], width,
                     color=colors[method], label=method)
        bar2 = ax.bar(pos_with, data[method]['with_h'], width,
                     color=colors[method], hatch='///', alpha=0.99)
        
        for j, val in enumerate(data[method]['without_h']):
            ax.text(pos_without[j], val, f'{val:.1f}', 
                   ha='center', va='bottom', fontsize=7)  # 调小字体
        for j, val in enumerate(data[method]['with_h']):
            ax.text(pos_with[j], val, f'{val:.1f}', 
                   ha='center', va='bottom', fontsize=7)  # 调小字体

    ax.set_ylabel('Performance', fontsize=10)  # 调整标签字体
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)  # 调整刻度字体
    ax.tick_params(axis='y', labelsize=8)  # 调整y轴刻度字体
    
    ax.set_ylim(0, 70)  # 调整y轴范围
    ax.set_yticks(np.arange(0, 71, 10))  # 调整刻度间隔
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend_elements = [
        plt.Rectangle((0,0),1,1, color=colors['SFT'], label='SFT'),
        plt.Rectangle((0,0),1,1, color=colors['ORL w/ SFT'], label='ORL w/ SFT'),
        plt.Rectangle((0,0),1,1, color=colors['ORL w/o SFT'], label='ORL w/o SFT'),
        plt.Rectangle((0,0),1,1, facecolor='gray', label='w/o Hierarchy'),
        plt.Rectangle((0,0),1,1, facecolor='gray', hatch='///', label='w/ Hierarchy')
    ]
    
    ax.legend(handles=legend_elements, ncol=5, bbox_to_anchor=(0.45, 1.07),
             loc='center', fontsize=8)  # 调小legend字体
    
    # ax.set_title('Ablation Study', pad=25, fontsize=10)  # 调整标题字体
    
    plt.tight_layout()
    return fig

fig = create_plot()
plt.show()

plt.savefig('ablation_study.png', dpi=1200, bbox_inches='tight')
plt.savefig('ablation_study.pdf', dpi=1200, bbox_inches='tight')