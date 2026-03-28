import torch
import torch.nn as nn

from decoder_layer import DecoderLayer
from position_wise_feed_forward import PositionWiseFeedForward
from transformer_embedding import transformerEmbedding

class decoder(nn.Module):
    def __init__(self, d_size, max_len , d_model , ffn_hidden , n_heads , n_layers , dropout , device):
        super(decoder, self).__init__()

        self.emb = transformerEmbedding(dim=d_model,
                                        dropout=dropout,
                                        max_len=max_len,
                                        vocab_size=d_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_heads=n_heads,dropout=dropout) 
                                     for _ in range(n_layers)])
    
        self.linear = nn.Linear(d_model, d_size)
    def forward(self, trg , enc_src , trg_mask , src_mask):
        
        x = self.emb(trg)

        for layer in self.layers:
            x = layer(x, enc_src, trg_mask, src_mask)
        
        out = self.linear(x)

        return out
