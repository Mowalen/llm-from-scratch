import torch
import torch.nn as nn
from posEncoding import posEncoding

class transformerEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, max_len, dropout, device):
        super(transformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = posEncoding(dim, max_len, device)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        x = token_emb + pos_emb 
        return self.dropout(x)