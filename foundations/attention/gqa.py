import torch
from torch import nn

class gqa(nn.Module):
    def __init__(self,hidden_dim,num_heads,nums_group,max_len,dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert num_heads % nums_group == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.nums_group = nums_group
        self.max_len = max_len
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_per_kv = num_heads // nums_group

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * num_heads)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)   

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        batch_size = x.shape[0]

        Q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        K = K.repeat_interleave(self.q_per_kv, dim=1)
        V = V.repeat_interleave(self.q_per_kv, dim=1)

        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            atten_scores = atten_scores.masked_fill(mask, float('-inf'))
        atten_scores = torch.softmax(atten_scores, dim=-1)
        atten_scores = self.dropout(atten_scores)

        out = (atten_scores @ V).transpose(1,2).reshape(batch_size, -1, self.hidden_dim)
        out = self.out_proj(out)

        return out


