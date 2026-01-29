import torch
import torch.nn as nn

class layerNorm(nn.Module):
    def __init__(self,hidden_dim,eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.bais = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True,unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.bais
    