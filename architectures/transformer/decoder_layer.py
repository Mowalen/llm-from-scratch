import torch
import torch.nn as nn
from attention import multihead_attention
from position_wise_feed_forward import PositionWiseFeedForward
from layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_dim,n_heads,dropout):   
        super().__init__()
        self.attention = multihead_attention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.droupout1 = nn.Dropout(dropout)

        self.enc_dec_attention = multihead_attention(d_model, n_heads)

        self.norm2 = LayerNorm(d_model)
        self.droupout2 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model, ffn_dim,dropout)
        self.norm3 = LayerNorm(d_model)
        self.droupout3 = nn.Dropout(dropout)

    def forward(self,dec,enc,trg_mask,src_mask):
        #dec: [batch_size, trg_len, d_model]
        #enc: [batch_size, src_len, d_model]
        
        #masked multi-head attention
        _dec = dec
        dec = self.attention(dec, dec, dec, trg_mask)
        dec = self.droupout1(dec)
        dec = self.norm1(dec + _dec)

        #encoder-decoder attention
        if enc is not None:
            _dec = dec
            dec = self.enc_dec_attention(dec, enc, enc, src_mask)
            dec = self.droupout2(dec)
            dec = self.norm2(dec + _dec)

        #position-wise feed forward
        _dec = dec
        dec = self.ffn(dec)
        dec = self.droupout3(dec)
        dec = self.norm3(dec + _dec)

        return dec
