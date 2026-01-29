from dataclasses import dataclass

@dataclass
class Qwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32 # 默认为多头注意力(MHA)，用户可以设置为分组查询注意力(GQA)
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    qkv_bias: bool = True # Qwen2 通常在 QKV 中使用偏置
