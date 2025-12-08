import torch
import torch.nn as nn

class lora_attention(nn.Module):
    
    def __init__(self, embed_dim, r):
        super(lora_attention, self).__init__()
        self.embed_dim = embed_dim
        self.r = r

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

        for param in self.W_Q.parameters():
            param.requires_grad = False
        for param in self.W_K.parameters():
            param.requires_grad = False
        for param in self.W_V.parameters():
            param.requires_grad = False

        # LoRA 的 Q 部分
        self.A_Q = nn.Parameter(torch.empty(r, embed_dim))
        self.B_Q = nn.Parameter(torch.zeros(embed_dim, r))
        nn.init.normal_(self.A_Q, mean=0.0, std=0.02)

        # LoRA 的 K 部分
        self.A_K = nn.Parameter(torch.empty(r, embed_dim))
        self.B_K = nn.Parameter(torch.zeros(embed_dim, r))
        nn.init.normal_(self.A_K, mean=0.0, std=0.02)

        # LoRA 的 V 部分
        self.A_V = nn.Parameter(torch.empty(r, embed_dim))
        self.B_V = nn.Parameter(torch.zeros(embed_dim, r))
        nn.init.normal_(self.A_V, mean=0.0, std=0.02)

def forward(self, query, key, value):
    """
    query, key, value: 形状为 (batch_size, seq_length, embed_dim)
    """
    Q = self.wq(query)  # (batch_size, seq_length, embed_dim)
    K = self.wk(key)
    V = self.wv(value)

    # 计算 LoRA 增量部分
    delta_Q = torch.matmul(query, self.A_Q.t())  # (batch_size, seq_length, r)
    delta_Q = torch.matmul(delta_Q, self.B_Q.t())  # (batch_size, seq_length, embed_dim)
    delta_K = torch.matmul(key, self.A_K.t())
    delta_K = torch.matmul(delta_K, self.B_K.t())
    delta_V = torch.matmul(value, self.A_V.t())
    delta_V = torch.matmul(delta_V, self.B_V.t())

    # 更新后的 Q、K、V
    Q = Q + delta_Q
    K = K + delta_K
    V = V + delta_V

    # 计算注意力得分
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, V)

    # 输出层
    output = self.W_O(context)

    return output