import torch
from safetensors.torch import load_file

# 路径
safetensors_path = "/root/autodl-tmp/sd-model/unet/diffusion_pytorch_model.non_ema.safetensors"
bin_path = "/root/autodl-tmp/sd-model/unet/diffusion_pytorch_model.bin"

print(f"正在把 {safetensors_path} 转换为 {bin_path} ...")

try:
    # 1. 读取 safetensors
    tensors = load_file(safetensors_path)
    
    # 2. 保存为 bin
    torch.save(tensors, bin_path)
    print(">>> 转换成功！")
except Exception as e:
    print(f">>> 报错了: {e}")