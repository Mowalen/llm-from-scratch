import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):

    def __init__(self, in_dim , out_dim , r , alpha,bias = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.r = r
        self.alpha = alpha
        self.scale = self.alpha / self.r

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.lora_a = nn.Linear(in_dim, r)
        self.lora_b = nn.Linear(r, out_dim)
        
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))

        nn.init.zeros_(self.lora_b.weight)  

        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

    def forward(self, x):

        original_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(x)) * self.scale
        return original_out + lora_out
    

