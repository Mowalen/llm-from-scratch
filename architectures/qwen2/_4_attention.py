import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from ._1_configuration import Qwen2Config

def rotate_half(x):
    """旋转输入的一半隐藏维度。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """将旋转位置嵌入应用于查询和键张量。"""
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim] 或 [1, 1, seq_len, head_dim]
    
    # 确保 cos/sin 形状适合广播
    # 如果 cos/sin 是 [seq_len, head_dim]，我们需要扩展/重塑它们
    # 通常为 [1, 1, seq_len, head_dim]，但我们要处理传入的情况。
    
    # 我们假设 cos 和 sin 已准备好可广播或者是 [seq_len, head_dim]
    # 如果它们只是 [seq_len, head_dim]，我们进行重塑。
    if len(cos.shape) == 2:
        cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 如果传递了 position_ids，我们可能需要选择特定的位置，但通常调用者已处理切片。
    # 这里我们执行逐元素乘法
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Qwen2 通常在 Q, K, V 投影中使用 bias=True
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False) # 输出通常在类 Llama 模型中没有偏置，但需要检查 Qwen。
        # 检查 Qwen2：o_proj 通常没有 bias。qkv 有 bias。

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # (cos, sin)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # RoPE 集成
        # 我们假设 freqs_cis (cos, sin) 已传入，并针对最大序列长度进行了预计算
        if freqs_cis is not None:
            cos, sin = freqs_cis
            # 将 cos/sin 切片到当前序列长度。
            # 注意：对于带缓存的推理，我们需要对应于新 token 的位置。
            # 这里仅作简单实现：假设非缓存的全序列或正确处理了 position_ids
            
            # 如果 position_ids 为 None，假设为 0..q_len
            if position_ids is None:
                 cos = cos[:q_len]
                 sin = sin[:q_len]
            else:
                 # 如果基本切片不够（例如左填充），需根据 position_ids 进行选择
                 # 简化：假设简单切片在此演示中有效
                 pass

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # 重用缓存
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 重复 K/V 头以用于 GQA
        # key_states: [bsz, num_kv_heads, seq_len, head_dim]
        # 我们需要重复它们以匹配 num_heads
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask 形状: [bsz, 1, q_len, kv_seq_len] 通常
            attn_weights = attn_weights + attention_mask

        # 转换为 float32 以保证 softmax 稳定性
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
