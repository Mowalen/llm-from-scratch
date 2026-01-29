import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self,hidden_dim,intermediate_dim,bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim,intermediate_dim,bias=bias)
        self.up_proj = nn.Linear(hidden_dim,intermediate_dim,bias=bias)
        self.down_proj = nn.Linear(intermediate_dim,hidden_dim,bias=bias)
    def forward(self,x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

