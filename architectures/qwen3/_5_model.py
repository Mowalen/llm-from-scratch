import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from ._1_configuration import Qwen3Config
from ._2_rms_norm import Qwen3RMSNorm
from ._3_moe import Qwen3MoE
from ._4_attention import Qwen3Attention
from .mlp import Qwen3MLP

def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()
    freqs_cis = (torch.cos(freqs), torch.sin(freqs))
    return freqs_cis

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)
        
        # 核心逻辑：决定是使用 Dense MLP 还是 MoE
        # 如果 decoder_sparse_step 为 1，则所有层都使用 MoE (Total MoE)
        # 如果 > 1，则间隔使用
        if config.decoder_sparse_step > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0:
            self.mlp = Qwen3MoE(config)
        else:
            self.mlp = Qwen3MLP(config)
            
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            freqs_cis=freqs_cis
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 无论是 MLP 还是 MoE，调用接口保持一致
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            config.max_position_embeddings, 
            config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        batch_size, seq_length = input_ids.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        inputs_embeddings = self.embed_tokens(input_ids)
        freqs_cis = (self.freqs_cis[0].to(inputs_embeddings.device), self.freqs_cis[1].to(inputs_embeddings.device))
        
        hidden_states = inputs_embeddings
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                freqs_cis=freqs_cis
            )
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        return hidden_states, next_decoder_cache

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重初始化略
        
    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states, past_key_values = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # 如果 MoE 使用了 load balancing loss (aux_loss)，这里应该加上
            # 这里的实现为了简化，暂未在 forward 返回 aux_loss
            
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": past_key_values
        }
