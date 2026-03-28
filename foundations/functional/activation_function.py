import torch
import numpy as np

def sigmod(self,x):
    return 1/(1+torch.exp(-x))


def softmax(self,x):
    exp_x = torch.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(self,x):
    return torch.maximum(torch.tensor(0.0), x)

def silu(self,x):
    return x * sigmod(x)