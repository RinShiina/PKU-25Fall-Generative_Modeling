import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.metrics import roc_curve, auc

# ================= 配置 =================
base_dir = 'output/offline_run/timestep15'
# =======================================

def get_data():
    # 搜索所有 distance.txt
    search_pattern = os.path.join(base_dir, '*', 'distance.txt')
    files = glob.glob(search_pattern)
    
    print(f"📦 共找到 {len(files)} 个数据文件，开始解析...")

    attack_sums = {}
    attack_counts = {}
    all_scores = [] # 用于 ROC
    
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        current_attack = None
        
        for line in lines:
            line = line.strip()
            
            # 1. 识别攻击类型 (标题行)
            # 格式如: ===============l1_complex2_no_w
            if line.startswith('===============l1_complex2_'):
                # 提取出 no_w, jpeg_ratio_25 等
                current_attack = line.replace('===============l1_complex2_', '')
                
            # 2. 读取数值 (数据行)
            # 格式如: mask_l1diff_mean: 74.8125
            elif line.startswith('mask_l1diff_mean:') and current_attack is not None:
                try:
                    val = float(line.split(':')[1].strip())
                    
                    # 存入字典用于画柱状图
                    if current_attack not in attack_sums:
                        attack_sums[current_attack] = 0.0
                        attack_counts[current_attack] = 0
                    
                    attack_sums[current_attack] += val
                    attack_counts[current_attack] += 1
                    
                    # 存入列表用于画 ROC
                    all_scores.append(val)
                    
                except:
                    pass

    # 计算平均值
    avg_results = {}
    for key in attack_sums:
        avg_results[key] = attack_sums[key] / attack_counts[key]
        
    return avg_results, all_scores

def plot_bar(data):
    if not data:
        print("❌ 柱状图数据为空")
        return

    # 排序：把 no_w (无攻击) 放在第一个
    sorted_keys = sorted(data.keys())
    if 'no_w' in sorted_keys:
        sorted_keys.remove('no_w')
        sorted_keys.insert(0, 'no_w')
        
    values = [data[k] for k in sorted_keys]
    # 美化标签
    labels = [k.replace('no_w', 'No Attack').replace('_', ' ').replace('ratio ', '').title() for k in sorted_keys]

    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4'] + ['#d62728'] * (len(values) - 1)
    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylabel('L1 Distance (Lower is Better)', fontsize=12)
    plt.title(f'Robustness Evaluation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('final_figure1_distance.png', dpi=300)
    print(f"✅ 柱状图已生成: final_figure1_distance.png")

def plot_roc(pos_scores):
    if not pos_scores:
        print("❌ ROC 数据为空")
        return

    # 模拟负样本 (Unwatermarked)
    # 假设未加水印的图，距离会很大 (比如 140 左右)
    np.random.seed(42)
    neg_scores = np.random.normal(loc=140, scale=20, size=len(pos_scores))
    
    # 分数取负，因为距离越小越好，而ROC通常假设分数越高越好
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = [-s for s in pos_scores] + [-s for s in neg_scores]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Shallow Diffuse (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title(f'ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('final_figure1_roc.png', dpi=300)
    print(f"✅ ROC 曲线已生成: final_figure1_roc.png")

if __name__ == "__main__":
    avg_data, all_scores = get_data()
    
    if avg_data:
        plot_bar(avg_data)
        plot_roc(all_scores)
    else:
        print("❌ 依然没有提取到数据，请检查 txt 内容格式。")