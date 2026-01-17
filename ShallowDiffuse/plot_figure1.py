import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
# 确保这个路径指向你的 distance.txt
target_dir = 'output/offline_run/timestep15/img0' 
file_path = os.path.join(target_dir, 'distance.txt')
# ===========================================

def parse_log_file(file_path):
    """解析日志格式的 distance.txt"""
    data = {}
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_attack = None
    target_section_found = False # 标记是否找到了 'l1_complex2_no_w' 这一段

    for line in lines:
        line = line.strip()
        
        # 1. 捕捉攻击名称，例如 **********none
        if line.startswith('****'):
            # 提取星号后面的名字
            parts = line.split('*')
            # 过滤空字符串，取最后一个非空的作为名字
            clean_parts = [p for p in parts if p]
            if clean_parts:
                current_attack = clean_parts[-1].strip()
                target_section_found = False # 换新攻击了，重置标记
        
        # 2. 锁定正确的数据块 (我们只看单图结果 l1_complex2_no_w，不看 avg)
        elif line.startswith('===============l1_complex2_no_w'):
            target_section_found = True
        
        # 3. 如果在正确的数据块里，提取数值
        elif target_section_found and line.startswith('mask_l1diff_mean:'):
            if current_attack:
                try:
                    val = float(line.split(':')[1].strip())
                    # 如果同一个攻击出现多次（比如日志追加了），保留最后一次，或者第一次
                    # 这里我们简单粗暴地覆盖，通常日志最后是新的
                    data[current_attack] = val
                except ValueError:
                    pass

    return data

def plot_chart(data):
    if not data:
        print("未提取到任何数据，请检查 txt 文件内容是否包含 'mask_l1diff_mean'")
        return

    # 排序：把 'none' 放在第一个作为基准
    attacks = sorted(list(data.keys()))
    if 'none' in attacks:
        attacks.remove('none')
        attacks.insert(0, 'none')
        
    distances = [data[k] for k in attacks]

    plt.figure(figsize=(12, 6))
    
    # 绘制柱状图
    # 注意：这里画的是 "距离 (Distance)"
    # 距离越小(柱子越短)，说明水印越接近原版，鲁棒性越好。
    # 距离越大(柱子越高)，说明水印被破坏得越厉害。
    bars = plt.bar(attacks, distances, color='salmon', edgecolor='darkred')
    
    # 在柱子上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 1), ha='center', va='bottom')

    plt.title('Watermark Distance under Attacks (Lower is Better)', fontsize=16)
    plt.xlabel('Attack Types', fontsize=12)
    plt.ylabel('L1 Distance (Mask Difference)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # 保存图片
    save_path = 'reproduced_figure1_distance.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    print(f"\n绘图完成！图片已保存为: {save_path}")
    print("-" * 30)
    print("【图表解读】")
    print("Y轴表示 'L1距离'。")
    print("柱子越【矮】，说明水印保留得越好（检测越容易）。")
    print("柱子越【高】，说明攻击破坏了水印（距离变大了）。")
    print("-" * 30)

if __name__ == "__main__":
    print(f"正在解析日志文件: {file_path} ...")
    data = parse_log_file(file_path)
    print("提取到的攻击数据:", data)
    plot_chart(data)