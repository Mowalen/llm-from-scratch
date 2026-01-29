import torch
import torch.nn as nn
from ._1_configuration import Qwen3Config

class Qwen3MLP(nn.Module):
    """
    标准的 MLP，用于非 MoE 层。
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        # 兼容旧配置及 MoE 配置逻辑，这里使用默认的 intermediate_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
