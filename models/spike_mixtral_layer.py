import torch
from torch import nn
from typing import Optional, Tuple, List
from spike_driven_quant.spike_linear import SpikeQuantLinear
from spike_driven_quant.spike_matmul import SpikeQuantMatMul
import torch.nn.functional as F
from spike_driven_quant.omni_norm import OmniLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import pdb
import copy
from models.transformation import *

# 确保我们能访问Mixtral的原始实现
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention, 
    MixtralSparseMoeBlock, 
    MixtralDecoderLayer
)

class SpikeMixtralAttention(nn.Module):
    def __init__(self, original_attn, args):
        super().__init__()
        self.args = args
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.head_dim = original_attn.head_dim
        self.rope_theta = original_attn.rope_theta
        
        # 量化线性层
        self.q_proj = SpikeLinear(original_attn.q_proj, args)
        self.k_proj = SpikeLinear(original_attn.k_proj, args)
        self.v_proj = SpikeLinear(original_attn.v_proj, args)
        self.o_proj = SpikeLinear(original_attn.o_proj, args)
        
        # 保持其他属性不变
        self.rotary_emb = original_attn.rotary_emb
        self.attn_implementation = original_attn.attn_implementation
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # 使用量化线性层投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 调整维度
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 处理past_key_value（用于缓存）
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # 应用RoPE位置编码
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0)
        
        if self.rotary_emb is not None:
            query_states = self.rotary_emb(query_states, position_ids)
            key_states = self.rotary_emb(key_states, position_ids)
        
        # 处理缓存的键和值（如果有）
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # 重复键值头以匹配查询头数（GQA实现）
        if self.num_key_value_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=0)
            value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=0)
        
        # 使用scaled_dot_product_attention计算注意力
        if self.attn_implementation == "eager":
            attn_weights = None  # 不保存注意力权重，除非需要输出
            
            # 调整查询头批次维度
            query_states = query_states.reshape(bsz * self.num_heads, q_len, self.head_dim)
            key_states = key_states.reshape(bsz * self.num_heads, -1, self.head_dim)
            value_states = value_states.reshape(bsz * self.num_heads, -1, self.head_dim)
            
            # 应用注意力掩码
            if attention_mask is not None:
                # 调整注意力掩码以适应bsz*num_heads的大小
                expanded_attn_mask = attention_mask.unsqueeze(1).expand(bsz, self.num_heads, -1, -1)
                expanded_attn_mask = expanded_attn_mask.reshape(bsz * self.num_heads, 1, -1)
                
                # 使用scaled_dot_product_attention
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=expanded_attn_mask,
                    is_causal=attention_mask is None,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    is_causal=True,
                )
            
            # 调整输出维度
            attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
            
        else:
            # 手动实现attention（如果不使用eager实现）
            raise NotImplementedError("只支持eager实现的注意力机制")
        
        # 调整输出维度
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # 应用输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value


class SpikeMixtralExpertMLP(nn.Module):
    def __init__(self, original_mlp, args):
        super().__init__()
        self.args = args
        
        # 量化三个线性层
        self.w1 = SpikeLinear(original_mlp.w1, args)
        self.w2 = SpikeLinear(original_mlp.w2, args)
        self.w3 = SpikeLinear(original_mlp.w3, args)
        
        # 保存激活函数
        self.act_fn = original_mlp.act_fn
    
    def forward(self, hidden_states):
        # 使用SwiGLU激活
        # 第一条路径：w1 -> act_fn
        w1_out = self.act_fn(self.w1(hidden_states))
        # 第二条路径：w3
        w3_out = self.w3(hidden_states)
        # 元素相乘
        expert_output = w1_out * w3_out
        # 最后的投影
        output = self.w2(expert_output)
        
        return output


class SpikeMixtralMoEBlock(nn.Module):
    def __init__(self, original_moe_block, args):
        super().__init__()
        self.args = args
        
        # 保留原始路由器，无需量化
        self.gate = original_moe_block.gate
        
        # 保存MoE配置
        self.num_experts = original_moe_block.num_experts
        self.top_k = original_moe_block.top_k
        
        # 量化每个专家
        self.experts = nn.ModuleList([
            SpikeMixtralExpertMLP(expert, args) for expert in original_moe_block.experts
        ])
    
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshape = hidden_states.view(-1, hidden_dim)
        
        # 使用路由器计算专家分配
        router_logits = self.gate(hidden_states_reshape)
        
        # 选择top-k专家
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # 创建结果tensor
        final_hidden_states = torch.zeros_like(hidden_states_reshape)
        
        # 处理每个专家
        flat_indices = torch.arange(selected_experts.shape[0], device=selected_experts.device)
        for expert_idx in range(self.num_experts):
            # 找出需要此专家处理的token位置
            expert_mask = (selected_experts == expert_idx)
            token_indices = flat_indices.unsqueeze(-1).expand_as(expert_mask)[expert_mask]
            
            if token_indices.shape[0] == 0:
                continue
            
            # 提取需要处理的隐藏状态
            expert_weights = routing_weights[expert_mask]
            expert_states = hidden_states_reshape[token_indices]
            
            # 通过专家处理
            processed_states = self.experts[expert_idx](expert_states)
            
            # 加权分配到最终结果
            final_hidden_states.index_add_(
                0, token_indices, processed_states * expert_weights.unsqueeze(-1)
            )
        
        # 重塑回原始维度
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class SpikeMixtralDecoderLayer(nn.Module):
    def __init__(self, original_layer, args):
        super().__init__()
        self.args = args
        
        # 量化自注意力层
        self.self_attn = SpikeMixtralAttention(original_layer.self_attn, args)
        
        # 量化MoE块
        self.block_sparse_moe = SpikeMixtralMoEBlock(original_layer.block_sparse_moe, args)
        
        # 保持LayerNorm不变（不量化）
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # 如果使用量化的LayerNorm，可以替换为：
        # self.input_layernorm = SpikeRMSNorm(original_layer.input_layernorm, args)
        # self.post_attention_layernorm = SpikeRMSNorm(original_layer.post_attention_layernorm, args)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        # 自注意力块 + 残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 自注意力
        attn_outputs, attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = residual + attn_outputs
        
        # MoE块 + 残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.block_sparse_moe(hidden_states)
        
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        if use_cache:
            outputs += (past_key_value,)
        
        return outputs


# 处理整个模型的函数
def create_spike_mixtral_model(model, args):
    """
    将原始Mixtral模型转换为量化版本
    
    Args:
        model: 原始Mixtral模型
        args: 量化参数
    
    Returns:
        量化后的模型
    """
    # 量化每个解码器层
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = SpikeMixtralDecoderLayer(layer, args)
    
    return model







