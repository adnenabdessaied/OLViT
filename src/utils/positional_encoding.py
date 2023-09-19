# https://github.com/pytorch/pytorch/issues/68407
from torch import nn
from torch import Tensor
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # positional encoding expects shape (seq_len, batch_size, emb_dim), (batch_size, seq_len, emb_dim) is given
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1,0,2)
        return self.dropout(x)

