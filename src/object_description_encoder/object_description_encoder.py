import pytorch_lightning as pl
from torch import nn
import torch


class ObjectDescriptionEncoder(pl.LightningModule):
    def __init__(self, d_model, config):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            batch_first=True,
            dropout=config['dropout_p'],
            dim_feedforward=config['object_feature_generator_dim'],
            nhead=config['n_heads']
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config['n_object_feature_generator_layers']
        )

    def forward(self, input):
        object_description_embedding = torch.zeros((input.shape[0], 1, self.d_model), device=self.device)
        input = torch.concat([object_description_embedding, input], dim=1)
        output = self.encoder(input)
        object_description_embedding = output[:, 0, :]
        object_description_embedding = object_description_embedding.unsqueeze(1)
        return object_description_embedding
    
