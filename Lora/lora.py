import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRALayer, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.r = r

        self.w = nn.Linear(in_features, out_features)
        self.w.requires_grad = False

        self.a = nn.Parameter(torch.empty(r, in_features))
        self.b = nn.Parameter(torch.zeros(out_features, r))

        nn.init.normal_(self.a, mean=0, std=0.02)

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):

        original_output = torch.nn.functional.linear(x, self.w.weight, self.w.bias)

        delta_w = torch.matmul(self.b, self.a)
        lora_output = torch.nn.functional.linear(x, delta_w)

        return original_output + lora_output
    



