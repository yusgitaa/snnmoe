import torch.nn as nn
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module.modeling_deepseek import DeepseekMoE, DeepseekConfig, SpikeDeepseekMoE


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
        act_layer=None,
        T=4,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
            
        # MoE配置
        self.use_moe = use_moe
        if use_moe:
            moe_config = DeepseekConfig(
                hidden_size=hidden_features,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                aux_loss_alpha=0.01,
                intermediate_size=hidden_features * 4,
                moe_intermediate_size=hidden_features * 4,
                hidden_dropout_prob=0.1,
                hidden_act="gelu",
                pretraining_tp=1,
                spike_mode=spike_mode,
                use_expert_residual=use_expert_residual,
            )
            self.moe = DeepseekMoE(moe_config)

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

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        # 原始代码路径，对非MoE模式
        if not hasattr(self, 'use_moe') or not self.use_moe:
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
            # MoE代码路径
            x = self.fc1_lif(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
            x = self.fc1_conv(x.flatten(0, 1))
            x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
            
            # 关键修复：保持残差连接的原始位置
            if self.res:
                x = identity + x
                identity = x
                
            # MoE只在启用时应用
            x = self.moe(x)
            
            x = self.fc2_lif(x)
            if hook is not None:
                hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
            x = self.fc2_conv(x.flatten(0, 1))
            x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

            x = x + identity
            return x, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        T=4
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

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
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
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
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
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v, hook


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        drop=0.,
        spike_mode="lif",
        use_moe_mlp=False,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        T=4,
    ):
        super().__init__()
        self.T = T
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn = MS_SSA_Conv(
            dim, 
            num_heads=num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=drop,
            sr_ratio=1,
            mode="direct_xor",
            spike_mode=spike_mode,
            dvs=False,
            layer=0,
            T=T
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if use_moe_mlp:
            self.mlp = SpikeDeepseekMoE(
                dim=dim,
                hidden_dim=mlp_hidden_dim,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                spike_mode=spike_mode,
                drop=drop,
                T=T,
                debug=False
            )
        else:
            self.mlp = MS_MLP_Conv(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=nn.GELU,
                drop=drop,
                spike_mode=spike_mode,
                T=T
            )

    def forward(self, x, hook=None):
        # 自注意力部分
        residual = x
        
        # 处理norm1 - 调整维度以适应LayerNorm
        T, B, C, H, W = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(-1, C)  # [T*B*H*W, C]
        x_norm = self.norm1(x_flat)  # 应用LayerNorm
        x = x_norm.reshape(T, B, H, W, C).permute(0, 1, 4, 2, 3)  # 恢复原始形状 [T, B, C, H, W]
        
        x, attn, hook = self.attn(x, hook)
        x = residual + x
        
        # MLP/MoE部分
        residual = x
        
        # 处理norm2 - 同样调整维度
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(-1, C)  # [T*B*H*W, C]
        x_norm = self.norm2(x_flat)  # 应用LayerNorm
        x = x_norm.reshape(T, B, H, W, C).permute(0, 1, 4, 2, 3)  # 恢复原始形状
        
        x = self.mlp(x)
        x = residual + x
        
        return x, attn, hook
