import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple
from module.modeling_deepseek import DeepseekMoE, DeepseekConfig

class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
        use_moe=False,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        T=4,
    ):
        super().__init__()
        self.T = T
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        if use_moe:
            moe_config = DeepseekConfig(
                hidden_size=embed_dims,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                aux_loss_alpha=0.01,
                intermediate_size=embed_dims * 4,
                moe_intermediate_size=embed_dims * 4,
                hidden_dropout_prob=0.1,
                hidden_act="gelu",
                pretraining_tp=1,
            )
            self.use_moe = use_moe
            self.moe = DeepseekMoE(moe_config)
        else:
            self.use_moe = False
            self.rpe_conv = nn.Conv2d(
                embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.rpe_bn = nn.BatchNorm2d(embed_dims)
            if spike_mode == "lif":
                self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
            elif spike_mode == "plif":
                self.rpe_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="torch"
                )

    def forward(self, x, hook=None):
        # 使用静态类变量来控制打印
        if hook is None:
            hook = {}
        
        # 初始化静态计数器（之前被移除的代码）
        if not hasattr(MS_SPS, '_print_count'):
            MS_SPS._print_count = 0
        
        # 仅在第一次调用时打印
        if MS_SPS._print_count == 0:
            print("\nSPS Debug Info:")
            print("1. Initial input shape:", x.shape)
            
            # 确保输入是5维的 [B, T, C, H, W]
            if len(x.shape) == 4:  # [B, C, H, W]
                B, C, H, W = x.shape
                print("2. Converting 4D input to 5D")
                x = x.unsqueeze(1)  # [B, 1, C, H, W]
                x = x.repeat(1, self.T, 1, 1, 1)  # [B, T, C, H, W]
            
            B, T, C, H, W = x.shape
            print(f"3. Shape before processing: [B={B}, T={T}, C={C}, H={H}, W={W}]")
        else:
            # 非首次调用时的维度处理
            if len(x.shape) == 4:  # [B, C, H, W]
                B, C, H, W = x.shape
                x = x.unsqueeze(1)  # [B, 1, C, H, W]
                x = x.repeat(1, self.T, 1, 1, 1)  # [B, T, C, H, W]
        
        # 处理输入
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        # 应用卷积层
        x = self.proj_conv(x)  # [B*T, 32, H, W]
        x = self.proj_bn(x)
        x = x.reshape(B, T, -1, H, W)
        x = self.proj_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        
        # 第一个maxpool
        x = x.reshape(B * T, -1, H, W)
        x = self.maxpool(x)
        H, W = H // 2, W // 2
        
        # 第二个卷积块
        x = self.proj_conv1(x)  # [B*T, 64, H/2, W/2]
        x = self.proj_bn1(x)
        x = x.reshape(B, T, -1, H, W)
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        
        # 第二个maxpool
        x = x.reshape(B * T, -1, H, W)
        x = self.maxpool1(x)
        H, W = H // 2, W // 2
        
        # 第三个卷积块
        x = self.proj_conv2(x)  # [B*T, 128, H/4, W/4]
        x = self.proj_bn2(x)
        x = x.reshape(B, T, -1, H, W)
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        
        # 第三个maxpool
        x = x.reshape(B * T, -1, H, W)
        x = self.maxpool2(x)
        H, W = H // 2, W // 2
        
        # 第四个卷积块
        x = self.proj_conv3(x)  # [B*T, 256, H/8, W/8]
        x = self.proj_bn3(x)
        x = self.maxpool3(x)
        H, W = H // 2, W // 2
        
        # MoE或RPE处理
        if self.use_moe:
            x = x.reshape(B, T, -1, H, W)  # [B, T, C, H, W]
            orig_shape = x.shape
            x = x.reshape(-1, x.shape[2])  # [B*T*H*W, C]
            
            x = self.moe(x)  # 使用 MoE
            
            # 重要：捕获 MoE 的辅助损失
            if hasattr(self.moe, 'aux_loss') and self.moe.aux_loss is not None:
                hook['moe_aux_loss'] = self.moe.aux_loss
            
            x = x.reshape(*orig_shape)  # 恢复原始形状
            
            if hook is not None:
                hook[self._get_name() + "_moe"] = x.detach()
        else:
            x_feat = x
            x = x.reshape(B, T, -1, H, W)
            x = self.rpe_lif(x)
            if hook is not None:
                hook[self._get_name() + "_lif3"] = x.detach()
            
            x = x.reshape(B * T, -1, H, W)
            x = self.rpe_conv(x)
            x = self.rpe_bn(x)
            x = x + x_feat
            x = x.reshape(B, T, -1, H, W)
        
        # 最后仅打印一次
        if MS_SPS._print_count == 0:
            print(f"4. Final output shape: {x.shape}")
            MS_SPS._print_count += 1
        
        return x, (H, W), hook
