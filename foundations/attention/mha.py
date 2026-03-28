import torch
import torch.nn as nn

class mha(nn.Module):
    def __init__(self,hidden_dim,num_heads,max_len,dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_len = max_len

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        batch_size = x.shape[0]

        Q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2) #(batch, num_heads, seq_len, head_dim)
        K = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale #(batch, num_heads, seq_len, seq_len)
        if mask is not None:
            atten_scores = atten_scores.masked_fill(mask, float('-inf'))
        atten_scores = torch.softmax(atten_scores, dim=-1)
        atten_scores = self.dropout(atten_scores)

        out = (atten_scores @ V).transpose(1,2).reshape(batch_size, -1, self.hidden_dim) #(batch, seq_len, hidden_dim)
        out = self.out_proj(out)
        return out