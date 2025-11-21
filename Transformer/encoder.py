import torch
import torch.nn as nn
from transformerEmbedding import transformerEmbedding
from encoderLayer import encoderLayer

class encoder(nn.Module):

    def __init__(self, e_size, max_len , d_model , ffn_hidden , n_heads , n_layers , dropout , device):
        super(encoder, self).__init__()
        self.emb = transformerEmbedding(dim=d_model,
                                        dropout=dropout,
                                        max_len=max_len,
                                        vocab_size=e_size,
                                        device=device)
        self.layers = nn.ModuleList([encoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_heads=n_heads,
                                                  dropout=dropout)
                                     for _ in range(n_layers)])
        

    def forward(self, x , src_mask):
        
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x