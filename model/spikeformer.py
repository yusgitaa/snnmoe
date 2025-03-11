from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dims=96,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=None,
        sr_ratios=None,
        T=4,
        pooling_stat="1111",
        spike_mode="lif",
        use_moe=False,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
            use_moe=use_moe,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode="direct_xor",
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", self.patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _, hook = patch_embed(x, hook=hook)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        # 仅在第一次调用时打印
        if not hasattr(SpikeDrivenTransformer, '_print_count'):
            SpikeDrivenTransformer._print_count = 0

        if SpikeDrivenTransformer._print_count == 0:
            print("\nSpikeformer Debug Info:")
            print("1. Input shape:", x.shape)
        
            # 确保输入是5维的 [B, T, C, H, W]
            if len(x.shape) == 4:  # [B, C, H, W]
                B = x.shape[0]
                x = x.unsqueeze(1)  # [B, 1, C, H, W]
                x = x.repeat(1, self.T, 1, 1, 1)  # [B, T, C, H, W]
                print("2. After time dimension:", x.shape)
            SpikeDrivenTransformer._print_count += 1
        else:
            # 非首次调用时的维度处理
            if len(x.shape) == 4:
                B = x.shape[0]
                x = x.unsqueeze(1)
                x = x.repeat(1, self.T, 1, 1, 1)
        
        x, hook = self.forward_features(x, hook=hook)
        if SpikeDrivenTransformer._print_count == 0:
            print("3. After forward_features:", x.shape)
        
        x = self.head_lif(x)
        if SpikeDrivenTransformer._print_count == 0:
            print("4. After head_lif:", x.shape)
        
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if SpikeDrivenTransformer._print_count == 0:
            print("5. After head:", x.shape)
        
        if not self.TET:
            x = x.mean(1)  # [B, features]
            if SpikeDrivenTransformer._print_count == 0:
                print("6. After mean:", x.shape)
                SpikeDrivenTransformer._print_count += 1
        
        aux_info = hook if hook is not None else {}

        # MoE 层处理
        if hasattr(self, 'moe') and self.moe is not None:
            if hasattr(self.moe, 'aux_loss') and self.moe.aux_loss is not None:
                aux_info['moe_aux_loss'] = self.moe.aux_loss
        
        return x, aux_info


@register_model
def sdt(
    pretrained=False,
    pretrained_cfg=None,
    T=4,
    num_heads=8,
    depths=4,
    mlp_ratios=4,
    sr_ratios=1,
    pooling_stat="1111",
    spike_mode="lif",
    use_moe=False,
    n_routed_experts=4,
    n_shared_experts=None,
    num_experts_per_tok=2,
    **kwargs,
):
    print(f"\nModel Debug - Init:")
    print(f"use_moe: {use_moe}")
    print(f"n_routed_experts: {n_routed_experts}")
    print(f"num_experts_per_tok: {num_experts_per_tok}")
    
    model_kwargs = dict(
        patch_size=4,
        embed_dims=256,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        qkv_bias=False,
        depths=depths,
        sr_ratios=sr_ratios,
        T=T,
        pooling_stat=pooling_stat,
        spike_mode=spike_mode,
        use_moe=use_moe,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    
    # 更新配置，允许通过kwargs覆盖默认值
    model_kwargs.update(kwargs)
    
    model = SpikeDrivenTransformer(**model_kwargs)
    model.default_cfg = _cfg()
    return model
