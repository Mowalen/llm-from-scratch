import torch
import torch.nn as nn
from ._1_configuration import Qwen2Config

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Qwen2 通常在 MLP 层中不使用偏置，但在 Attention 中使用。
        # 让我们验证一下：Qwen2-7B 配置通常为 `mlp_bias: False` (类似 Llama 的默认值)。
        # 除非另有说明，我们将假设 MLP 的 bias=False 以符合标准的现代实践。
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
