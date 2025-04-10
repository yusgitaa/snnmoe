import torch.nn as nn
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module.modeling_deepseek import DeepseekMoE, DeepseekConfig
import os
import torch
import numpy as np
from pathlib import Path
import math


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
        use_moe=False,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        use_expert_residual=False,
        aux_loss_alpha=0.01,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layer = layer
        
        # 使用MoE模式还是原始SNN模式
        self.use_moe = use_moe
        
        # 原始两层SNN模式的组件
        if not use_moe:
            self.res = in_features == hidden_features
            self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
            self.fc1_bn = nn.BatchNorm2d(hidden_features)
            if spike_mode == "lif":
                self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.fc1_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
                
            self.fc2_conv = nn.Conv2d(
                hidden_features, out_features, kernel_size=1, stride=1
            )
            self.fc2_bn = nn.BatchNorm2d(out_features)
            if spike_mode == "lif":
                self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.fc2_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
        
        # MoE模式 - 直接替代两层SNN
        else:
            # MoE配置
            moe_config = DeepseekConfig(
                hidden_size=in_features,  # 输入维度与in_features匹配
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                aux_loss_alpha=aux_loss_alpha,
                intermediate_size=hidden_features,  # 中间层维度
                moe_intermediate_size=hidden_features,  # 确保MoE中间层维度正确
                hidden_dropout_prob=drop,
                hidden_act="gelu",
                pretraining_tp=1,
                spike_mode=spike_mode,
                use_expert_residual=use_expert_residual,
                use_soft_prompt=True,  # 启用软提示
            )
            self.moe = DeepseekMoE(moe_config)
            
            # 打印关键参数以进行调试
            print(f"\nMS_MLP_Conv初始化MoE模式:")
            print(f"in_features: {in_features}")
            print(f"hidden_features: {hidden_features}")
            print(f"out_features: {out_features}")
            print(f"n_routed_experts: {n_routed_experts}")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x, orig_images=None, hook=None, epoch=None, batch_idx=None, img_indices=None):
        T, B, C, H, W = x.shape
        identity = x

        # 原始SNN模式
        if not self.use_moe:
            x = self.fc1_lif(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
            x = self.fc1_conv(x.flatten(0, 1))
            x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
            if self.res:
                x = identity + x
                identity = x
            x = self.fc2_lif(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
            x = self.fc2_conv(x.flatten(0, 1))
            x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

            x = x + identity
            return x, hook
        else:
            # MoE直接处理输入但不应用外部残差连接
            moe_output = self.moe(x)
            moe_output = moe_output + identity
            return moe_output, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.q_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.k_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.mode = mode
        self.layer = layer
        self.sr_ratio = sr_ratio
        
        # 默认关闭attention map保存
        self.save_attention = False
        self.attention_dir = None
        
    def visualize_attention_inference(self, x, attn_weights, orig_images, batch_idx, img_idx, layer_name):
        """推理时的注意力图可视化方法"""
        import matplotlib.pyplot as plt
        import torch.nn.functional as F
        
        # 处理原始图像
        if len(orig_images.shape) == 4:  # [B, C, H, W]
            orig_img = orig_images.detach().cpu()
        elif len(orig_images.shape) == 5:  # [T, B, C, H, W]
            orig_img = orig_images[0].detach().cpu()  # 使用第一个时间步
        
        # 归一化原始图像到 [0, 1]
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        
        # 处理注意力权重
        attn = attn_weights.detach().cpu()  # [num_heads, H*W]
        num_heads = attn.shape[0]
        
        # 获取原始图像的空间维度
        _, _, H, W = orig_img.shape
        
        # 确保处理所有注意力头
        for head_idx in range(num_heads):
            # 获取当前头的注意力图
            head_attn = attn[head_idx]  # [32]
            
            # 检查注意力值是否都相同
            if head_attn.max() - head_attn.min() < 1e-6:
                continue
            
            # 将一维注意力权重重塑为4x8的形状
            head_attn = head_attn.view(4, 8)  # [4, 8]
            
            # 添加批次和通道维度用于插值
            head_attn = head_attn.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 8]
            
            # 使用双线性插值调整到原始图像大小
            head_attn = F.interpolate(
                head_attn,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            # 移除批次和通道维度
            head_attn = head_attn.squeeze()  # [H, W]
            
            # 归一化注意力图到 [0, 1]
            epsilon = 1e-6
            head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min() + epsilon)
            
            # 创建可视化
            plt.figure(figsize=(15, 5))  # 增加图像大小
            
            # 原始图像 - 保持彩色
            plt.subplot(131)
            if orig_img.shape[1] == 3:
                plt.imshow(orig_img.permute(0, 2, 3, 1).squeeze())
            else:
                img_3ch = orig_img.repeat(1, 3, 1, 1)
                plt.imshow(img_3ch.permute(0, 2, 3, 1).squeeze())
            plt.title('Original Image', fontsize=12, pad=10)
            plt.axis('off')
            
            # 注意力图 - 调整colormap和显示参数
            plt.subplot(132)
            # 可以尝试以下几种colormap:
            # 'inferno' - 黄橙红渐变，对比度高
            # 'magma' - 紫红黄渐变，看起来更平滑
            # 'viridis' - 蓝绿黄渐变，科技感强
            # 'plasma' - 紫橙黄渐变，非常醒目
            attention_map = plt.imshow(head_attn, cmap='inferno')
            """ plt.colorbar(attention_map, fraction=0.046, pad=0.04)  # 添加颜色条 """
            plt.title(f'Attention Map (Head {head_idx})', fontsize=12, pad=10)
            plt.axis('off')
            
            # 叠加效果 - 调整透明度和颜色
            plt.subplot(133)
            if orig_img.shape[1] == 3:
                plt.imshow(orig_img.permute(0, 2, 3, 1).squeeze())
            else:
                img_3ch = orig_img.repeat(1, 3, 1, 1)
                plt.imshow(img_3ch.permute(0, 2, 3, 1).squeeze())
            
            # 调整叠加参数
            overlay = plt.imshow(head_attn, cmap='inferno', alpha=0.4)  # 增加透明度，使原图更清晰
            """ plt.colorbar(overlay, fraction=0.046, pad=0.04) """
            plt.title(f'Overlay (Head {head_idx})', fontsize=12, pad=10)
            plt.axis('off')
            
            # 调整整体布局
            plt.tight_layout(pad=3.0)
            
            # 保存图像，添加层标识
            save_path = os.path.join(self.attention_dir, f'attention_{layer_name}_img{img_idx}_head{head_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 增加DPI提高图像质量
            plt.close()
            
            print(f"已保存注意力图：{save_path}")

    def forward(self, x, orig_images=None, hook=None, epoch=None, batch_idx=None, img_indices=None):
        T, B, C, H, W = x.shape
        N = H * W
        identity = x
        
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        kv = k.mul(v)
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        
        # 修改注意力图保存逻辑
        if self.save_attention and orig_images is not None:
            for b in range(B):
                # 获取所有时间步的注意力权重
                attn = kv[:, b].detach()  # [T, num_heads, H*W, 1]
                attn = attn.mean(0)  # [num_heads, H*W, 1]
                attn = attn.squeeze(-1)  # [num_heads, H*W]
                img_idx = img_indices[b] if img_indices is not None else b
                
                # 添加层标识到文件名
                layer_name = self._get_name() + str(self.layer)
                self.visualize_attention_inference(x, attn, orig_images[b:b+1], b, img_idx, layer_name)
        
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, kv, hook


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        use_moe_mlp=False,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        use_expert_residual=False,
        aux_loss_alpha=0.01,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
            use_moe=use_moe_mlp,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
            use_expert_residual=use_expert_residual,
            aux_loss_alpha=aux_loss_alpha,
        )

    def forward(self, x, orig_images=None, hook=None, epoch=None, batch_idx=None, img_indices=None):
        x_attn, attn, hook = self.attn(
            x, 
            orig_images=orig_images, 
            hook=hook, 
            epoch=epoch, 
            batch_idx=batch_idx,
            img_indices=img_indices
        )
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook


def visualize_attention(model, images, labels, save_dir, config, num_samples=10):
    """推理时的注意力图可视化函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 选择要可视化的样本
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for idx in indices:
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
            # 开启注意力图保存
            for module in model.modules():
                if hasattr(module, 'save_attention'):
                    module.save_attention = True
                    module.attention_dir = str(save_dir)
            
            # 前向传播，传递img_indices参数
            output, _ = model(image, orig_images=orig_image, img_indices=[idx])
            
            # 处理输出维度
            if isinstance(output, tuple):
                output = output[0]
            
            # 获取预测结果
            if output.dim() > 2:
                output = output.mean(0)
            pred = output.argmax(dim=-1).item()
        
        print(f"已处理样本 {idx}，标签: {label}，预测: {pred}")
