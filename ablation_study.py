import numpy as np
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x = np.arange(len(models)) * 0.4  # 缩小模型间距
    width = 0.06
    
    colors = {
        'SFT': '#8884d8',
        'ORL w/ SFT': '#82ca9d',
        'ORL w/o SFT': '#ffc658'
    }
    
    for i, method in enumerate(methods):
        pos = x + width * (i * 2 - 2.5)
        bars1 = ax.bar(pos, data[method]['without_h'], width,
                      label=method, color=colors[method])
        
        pos = x + width * (i * 2 - 1.5)
        bars2 = ax.bar(pos, data[method]['with_h'], width,
                      label=f'H-{method}', color=colors[method],
                      hatch='/')
        
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        add_labels(bars1)
        add_labels(bars2)

    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    
    # 调整y轴刻度
    ax.set_ylim(0, 70)
    ax.set_yticks(np.arange(0, 71, 10))  # 每10个单位一个刻度
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 调整legend和标题位置
    ax.legend(ncol=6, bbox_to_anchor=(0.5, 1.1), loc='center', fontsize=8)
    ax.set_title('Ablation Study', pad=5)
    
    plt.tight_layout()
    
    return fig

fig = create_plot()
plt.show()
plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
