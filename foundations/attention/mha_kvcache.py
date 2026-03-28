import torch.nn as nn
import torch

class mha_kvcache(nn.Module):

    def __init__(self,hidden_dim,num_heads,max_len,dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_len = max_len

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None,past_kv_value=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv_value is not None:
            past_len = past_kv_value[0].shape[2]

        Q = self.rope(Q, offset=past_len)
        K = self.rope(K, offset=past_len)

        if past_kv_value is not None:
            past_key, past_value = past_kv_value
            K = torch.cat([past_key, K], dim=2)
            V = torch.cat([past_value, V], dim=2)
        
        present_kv_value = (K, V)

        attn_scores = Q @ K.transpose(-1, -2) * self.scale
        if mask:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        output = (attn_scores @ V).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)

        return output, attn_scores, present_kv_value