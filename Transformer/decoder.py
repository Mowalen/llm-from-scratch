import torch
import torch.nn as nn

from decoderLayer import decoderLayer
from positionWiseFeedForward import positionWiseFeedForward
from transformerEmbedding import transformerEmbedding

class decoder(nn.Module):
    def __init__(self, d_size, max_len , d_model , ffn_hidden , n_heads , n_layers , dropout , device):
        super(decoder, self).__init__()

        self.emb = transformerEmbedding(dim=d_model,
                                        dropout=dropout,
                                        max_len=max_len,
                                        vocab_size=d_size,
                                        device=device)

        self.layers = nn.ModuleList([decoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_heads=n_heads,dropout=dropout) 
                                     for _ in range(n_layers)])
    
        self.linear = nn.Linear(d_model, d_size)
    def forward(self, trg , enc_src , trg_mask , src_mask):
        
        x = self.emb(trg)

        for layer in self.layers:
            x = layer(x, enc_src, trg_mask, src_mask)
        
        out = self.linear(x)

        return out