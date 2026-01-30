import torch
from torch import nn

class rope(nn.Module):
    def __init__(self,head_dim,max_len,base = 10000):
        super().__init__()

        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base
        self.max_len = max_len

        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos_id = torch.arange(max_len).float()
        freqs = torch.outern(pos_id, theta)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        self.register_buffer("freqs_sin", sin)
        self.register_buffer("freqs_cos", cos)

    def forward(self, x , offset = 0):
        _ , _ ,seq_len, _ = x.shape
        sin = self.freqs_sin[offset:offset+seq_len]
        cos = self.freqs_cos[offset:offset+seq_len]

        #将向量的最后一个维度（head_dim）按照“偶数位”和“奇数位”拆分开
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 使用 stack 和 flatten/reshape 来高效地交错合并
        # 1. 堆叠: [batch_size, num_heads, seq_len, head_dim / 2, 2]
        # 2. 展平: [batch_size, num_heads, seq_len, head_dim]        
        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
        return rotated_x
