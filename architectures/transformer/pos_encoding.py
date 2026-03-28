import torch
import math
import torch.nn as nn


class posEncoding(nn.Module):
    def __init__(self, d_model, max_len,device):
        super(posEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len,device).float().unsqueeze(1)

        _2i = torch.arange(0, d_model, 2,device).float()

        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))



    def forward(self, x):

        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]