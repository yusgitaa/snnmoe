import os
from datasets import load_dataset

# === 禁用代理 ===
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ[var] = "http://127.0.0.1:7890"

# === 下载数据集 ===
dataset = load_dataset("c4", "en", split="train[:1%]")
