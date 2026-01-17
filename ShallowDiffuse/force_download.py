import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
from huggingface_hub import snapshot_download

print(">>> 开始下载到数据盘...")
try:
    snapshot_download(
        repo_id='runwayml/stable-diffusion-v1-5',
        local_dir='/root/autodl-tmp/sd-model',  # <--- 注意这里改成了数据盘路径
        ignore_patterns=['*.ckpt'],
        resume_download=True,
        max_workers=4
    )
    print(">>> 恭喜！模型下载完成！")
except Exception as e:
    print(f">>> 报错了: {e}")