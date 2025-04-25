import os

# 设置是否禁用代理
DISABLE_PROXY = False

if DISABLE_PROXY:
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ[var] = ""

from huggingface_hub import snapshot_download

# ==== 模型信息 ====
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

target_dir = "./Mixtral-8x7B-Instruct-v0.1"

# ==== 下载 ====
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,  # 避免断点重下
    max_workers=4          # 可选：限制线程数，防止网络不稳
)
