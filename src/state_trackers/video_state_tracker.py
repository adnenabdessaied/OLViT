import pytorch_lightning as pl
from torch import nn
import torch


class VstLSTM(pl.LightningModule):
    def __init__(self, emb_dim, dropout, config):
        super().__init__()
        self.lstm_layer = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=config['num_layers_v_state'],
            batch_first=True,
            dropout=dropout
        )
        self.h = None
        self.c = None
 
    def forward(self, input):
        if self.h is None:
            _, (self.h, self.c) = self.lstm_layer(input)
        else:
            _, (self.h, self.c) = self.lstm_layer(input, (self.h, self.c))

        output = torch.permute(self.h, (1,0,2))
        output = output[:, -1, :]
        output = output.unsqueeze(1)
        return output

    def reset(self):
        self.h = None
        self.c = None



        