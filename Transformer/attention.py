import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None,e = 1e-12):
        batch_size,head,len_q,len_k = k.size()
        
        #[batch_size, head, length, d_tensor]
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32) + e)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
class multihead_attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.attention = ScaleDotProductAttention()
    def _split_heads(self, x):
        # param tensor: [batch_size, length, d_model]
        # return: [batch_size, head, length, d_tensor]
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 

    def forward(self, q, k, v, mask=None):
        q,k,v = self.linear_q(q),self.linear_k(k),self.linear_v(v)
        q,k,v = self._split_heads(q),self._split_heads(k),self._split_heads(v)
        output, attn_weights = self.attention(q, k, v, mask)

        out = self.concat_heads(output)
        out = self.linear_out(out)

        return out
    def concat_heads(self, x):
        # param tensor: [batch_size, head, length, d_tensor]
        # return: [batch_size, length, d_model]
        batch_size, head, seq_len, d_tensor = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
