import torch
from safetensors.torch import load_file

# 路径
safetensors_path = "/root/autodl-tmp/sd-model/vae/diffusion_pytorch_model.safetensors"
bin_path = "/root/autodl-tmp/sd-model/vae/diffusion_pytorch_model.bin"

print(f"正在把 {safetensors_path} 转换为 {bin_path} ...")

try:
    tensors = load_file(safetensors_path)
    torch.save(tensors, bin_path)
    print(">>> 转换成功！")
except Exception as e:
    print(f">>> 报错了: {e}")