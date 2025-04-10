""" python inference_visualize.py -c conf/cifar10/2_256_300E_t4.yml --model sdt 
--spike-mode lif --use-moe-mlp --n-shared-experts 1 --n-routed-experts 3 --num-experts-per-tok 1 推理代码"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from model.spikeformer import sdt
from module.ms_conv import MS_SSA_Conv
import argparse
import yaml
import numpy as np
import os
import math

def parse_args():
    parser = argparse.ArgumentParser(description='SpikeFormer Inference')
    
    # 基础参数
    parser.add_argument('-c', '--config', type=str, 
                       default='conf/cifar10/2_256_300E_t4.yml',
                       help='配置文件路径')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='sdt',
                       help='模型类型')
    parser.add_argument('--spike-mode', type=str, default='lif',
                       help='脉冲模式')
    parser.add_argument('--use-moe-mlp', action='store_true',
                       help='是否使用MoE MLP')
    parser.add_argument('--n-shared-experts', type=int, default=1,
                       help='共享专家数量')
    parser.add_argument('--n-routed-experts', type=int, default=3,
                       help='路由专家数量')
    parser.add_argument('--num-experts-per-tok', type=int, default=1,
                       help='每个token的专家数量')
    
    # 推理特有参数
    parser.add_argument('--model-path', type=str, 
                       default='output/train/20250409-130555-sdt-data-cifar10-t-4-spike-lif/model_best.pth.tar',
                       help='模型权重路径')
    parser.add_argument('--save-dir', type=str, 
                       default='inference_results',
                       help='结果保存目录')
    parser.add_argument('--num-samples', type=int, 
                       default=10,
                       help='要可视化的样本数量')
    
    args = parser.parse_args()
    return args

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def visualize_attention(model, images, labels, save_dir, config, num_samples=10):
    """推理时的注意力图可视化函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 使用前num_samples个样本，而不是随机选择
    for idx in range(min(num_samples, len(images))):
        # 获取图像和标签
        image = images[idx].unsqueeze(0)  # [1, 3, 32, 32]
        label = labels[idx]
        
        # 添加时间维度 T=4
        image = image.unsqueeze(0).repeat(4, 1, 1, 1, 1)  # [T, B, C, H, W]
        image = image.to(device)
        
        # 保存原始图像用于可视化
        orig_image = image.clone()
        
        # 推理
        with torch.no_grad():
            # 开启注意力图保存并设置保存目录
            for name, module in model.named_modules():
                if isinstance(module, MS_SSA_Conv):
                    module.save_attention = True
                    module.attention_dir = str(save_dir)
                    print(f"找到注意力模块 {name}，已启用注意力图保存")
            
            # 前向传播，传递img_indices参数
            output, _ = model(image, orig_images=orig_image, img_indices=[idx])
            
            # 处理输出维度
            if isinstance(output, tuple):
                output = output[0]
            
            # 确保输出是2D张量 [batch_size, num_classes]
            if output.dim() > 2:
                output = output.mean(0)  # [B, num_classes]
            
            # 获取预测结果
            if output.dim() == 2:
                pred = output[0].argmax().item()
            else:
                pred = output.argmax().item()
        
        print(f"已处理样本 {idx}，标签: {label}，预测: {pred}")

def main():
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 创建模型 - 使用命令行参数
    model = sdt(
        T=config['time_steps'],
        num_heads=config['num_heads'],
        depths=config['layer'],
        mlp_ratios=config['mlp_ratio'],
        spike_mode=args.spike_mode,
        use_moe_mlp=args.use_moe_mlp,
        n_shared_experts=args.n_shared_experts,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        num_classes=config['num_classes']
    )
    
    # 检查模型权重文件是否存在
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误：找不到模型权重文件 {model_path}")
        print("请确保模型权重文件存在，并正确指定路径")
        return
    
    try:
        # 加载权重
        checkpoint = torch.load(str(model_path), map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"成功加载模型权重：{model_path}")
    except Exception as e:
        print(f"加载模型权重时出错：{e}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 加载测试集 - 使用配置文件中的参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'],
                           std=config['std'])
    ])
    
    # 根据数据集名称加载对应的数据集
    if config['dataset'] == 'torch/cifar10':
        testset = torchvision.datasets.CIFAR10(
            root=config['data_dir'], 
            train=False,
            download=True, 
            transform=transform
        )
    elif config['dataset'] == 'torch/cifar100':
        testset = torchvision.datasets.CIFAR100(
            root=config['data_dir'], 
            train=False,
            download=True, 
            transform=transform
        )
    else:
        raise ValueError(f"不支持的数据集: {config['dataset']}")
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=config['val_batch_size'],
        shuffle=False, 
        num_workers=config['workers']
    )
    
    # 获取一批测试数据
    images, labels = next(iter(testloader))
    
    # 可视化注意力图
    visualize_attention(
        model,
        images,
        labels,
        args.save_dir,
        config,  # 传递config参数
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()