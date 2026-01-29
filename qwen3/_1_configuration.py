from dataclasses import dataclass
from typing import Optional

@dataclass
class Qwen3Config:
    vocab_size: int = 152064
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    qkv_bias: bool = True
    
    # MoE (Mixture of Experts) 配置 - Qwen3 核心升级
    # Qwen-MoE 特点：细粒度专家 + 共享专家
    decoder_sparse_step: int = 1 # moE 层的频率，1表示每层都是MoE，或者指定特定层
    moe_intermediate_size: int = 1408 # 专家的维度通常比 Dense MLP 小
    num_experts: int = 60      # 总路由专家数
    num_experts_per_tok: int = 4 # 每个 token 激活的路由专家数
    num_shared_experts: int = 4 # 共享专家数 (总是激活)
    # 共享专家中间维度，如果为None则由 moe_intermediate_size * num_shared_experts 决定
    # 但通常为了灵活控制，也会单独设定
    shared_expert_intermediate_size: int = 5632 
    
    # 路由与负载均衡相关
    router_aux_loss_coef: float = 0.001
    sequence_parallel_size: int = 1
