import torch
import platform

print(f"Python版本: {platform.python_version()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else '不可用'}")
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前设备索引: {torch.cuda.current_device()}")
    print(f"设备能力: {torch.cuda.get_device_capability(0)}")
    print(f"设备属性:")
    props = torch.cuda.get_device_properties(0)
    for prop in dir(props):
        if not prop.startswith('_'):
            print(f"  {prop}: {getattr(props, prop)}") 