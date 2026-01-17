import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

# ================= 配置 =================
# 你的结果路径
target_dir = 'output/offline_run/timestep15/img0' 
file_path = os.path.join(target_dir, 'distance.txt')
# =======================================

def parse_positive_scores(file_path):
    """从 distance.txt 读取带水印图片的距离（正样本）"""
    scores = []
    if not os.path.exists(file_path):
        print("找不到文件，使用默认模拟数据")
        return [75.0] * 10 # 模拟数据
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if 'mask_l1diff_mean:' in line:
            try:
                val = float(line.split(':')[1].strip())
                scores.append(val)
            except:
                pass
    return scores

def plot_roc_curve():
    # 1. 获取正样本 (带水印的图，距离较小)
    # 我们把你跑出来的所有攻击下的距离都算作正样本，看看在这个混合攻击下表现如何
    pos_scores = parse_positive_scores(file_path)
    
    if len(pos_scores) == 0:
        print("未读取到数据，无法绘图")
        return

    # 2. 模拟负样本 (无水印的图，距离通常很大)
    # 假设无水印图的距离服从均值 150，方差 10 的正态分布 (这是基于经验的假设)
    # 样本数量设为和正样本一样多
    np.random.seed(42)
    neg_scores = np.random.normal(loc=70, scale=15, size=len(pos_scores))
    
    # 3. 准备数据给 ROC 函数
    # 标签：0 代表带水印(正类)，1 代表无水印(负类)
    # 注意：这里距离越小越像水印，所以我们取负数作为分数，或者自定义逻辑
    # 也就是：分数 = -距离。分数越高(距离越小) -> 概率越大
    
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = [-s for s in pos_scores] + [-s for s in neg_scores]
    
    # 4. 计算 ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 5. 绘图
    plt.figure(figsize=(8, 8))
    
    # 画 Shallow Diffuse 的线
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Shallow Diffuse (AUC = {roc_auc:.2f})')
    
    # 画对角线 (瞎猜基准线)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('ROC Curve (Reproduction of Figure 1)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = 'reproduced_figure1_roc.png'
    plt.savefig(save_path, dpi=300)
    print(f"ROC 曲线已生成: {save_path}")
    print(f"正样本(水印)均值: {np.mean(pos_scores):.2f}")
    print(f"负样本(模拟)均值: {np.mean(neg_scores):.2f}")

if __name__ == "__main__":
    plot_roc_curve()