import pytorch_lightning as pl
from torch import nn
import torch


class DstTransformer(pl.LightningModule):
    def __init__(self, emb_dim, dropout, config):
        super().__init__()
        self.emb_dim = emb_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=config['dim_feedforward_d_transformer'],
            nhead=config['n_heads_state_tracker']
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config['num_layers_d_state']
        )
        self.state_vector = None

 
    def forward(self, input):
        if self.state_vector is None:
            self.state_vector = torch.zeros((input.shape[0], 1, self.emb_dim), device=self.device)
        
        input = torch.concat([self.state_vector, input], dim=1)
        output = self.encoder(input)
        self.state_vector = output[:, 0, :]
        self.state_vector = self.state_vector.unsqueeze(1)
        return self.state_vector


    def reset(self):
        self.state_vector = None
