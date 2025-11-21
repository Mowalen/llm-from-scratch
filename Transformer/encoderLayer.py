import torch
import torch.nn as nn
from attention import multihead_attention
from positionWiseFeedForward import PositionWiseFeedForward
from layerNorm import LayerNorm


class encoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_dim, n_heads, dropout):   
        super().__init__()
        self.attention = multihead_attention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model, ffn_dim, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src: [batch_size, src_len, d_model]
        
        # multi-head attention
        _src = src
        src = self.attention(src, src, src, src_mask)
        src = self.dropout1(src)
        src = self.norm1(src + _src)

        # position-wise feed forward
        _src = src
        src = self.ffn(src)
        src = self.dropout2(src)
        src = self.norm2(src + _src)

        return src