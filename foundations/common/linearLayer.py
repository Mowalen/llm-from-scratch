import torch
from torch import nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.rand(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.rand(output_dim))
        else:
            self.bias = None


    def forward(self, x):
        output = x @ self.weight.t()
        if self.bias is not None:
            output += self.bias
        return output