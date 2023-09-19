import pytorch_lightning as pl
import torch
from torch import nn

class CombinerOptionC(pl.LightningModule):
    def __init__(self, config, model_input_dim, use_v_state, use_d_state):
        super().__init__()
        self.config = config
        self.use_v_state = use_v_state
        self.use_d_state = use_d_state

        self.encoder_layer_d = nn.TransformerEncoderLayer(
            d_model=model_input_dim,
            dim_feedforward=self.config['dim_feedforward_d_transformer'],
            batch_first=True,
            nhead=self.config['n_heads_combiner_transformer']
        )
        self.encoder_layer_v = nn.TransformerEncoderLayer(
            d_model=model_input_dim,
            dim_feedforward=self.config['dim_feedforward_v_transformer'],
            batch_first=True,
            nhead=self.config['n_heads_combiner_transformer']
        )


    def prepare_inputs_for_transformers(self, video_emb, language_emb, language_emb_mask, v_state, d_state):
        # create masks for the language inputs (video seq should all be 301 frames long and dont need padding)
        d_input_mask = ~language_emb_mask # emb for pytorch needs to be True for masked tokens (opposite to huggingface mask)
        # if the dialogue state is used, add a column of Falses at the beeginngin of the tensor (state should be attended -> no mask)  
        if d_state is not None and self.use_d_state:
            zero_column = torch.zeros((d_input_mask.shape[0], 1), dtype=torch.bool, device=self.device)
            d_input_mask = torch.concat([zero_column, d_input_mask],axis=1)

        # prepare the input tensors for the different transformer layers depending on which state vectors should be used
        if v_state is not None \
             and d_state is not None \
             and self.use_v_state \
             and self.use_d_state:
            v_input = torch.concat([v_state, video_emb], axis=1)
            d_input = torch.concat([d_state, language_emb], axis=1)
        elif d_state is not None and self.use_d_state:
            v_input = video_emb
            d_input = torch.concat([d_state, language_emb], axis=1)
        elif v_state is not None and self.use_v_state:
            v_input = torch.concat([v_state, video_emb], axis=1)
            d_input = language_emb
        else:
            v_input = video_emb
            d_input = language_emb

        return v_input, d_input, d_input_mask


    def forward(self, dummy_word, video_emb, language_emb, language_emb_mask, v_state, d_state):
        # prepare the input tensors for the different transformer layers depending on which state vectors should be used
        v_input, d_input, d_input_mask = self.prepare_inputs_for_transformers(video_emb, language_emb, language_emb_mask, v_state, d_state)

        # apply the v transformer to the v input and the d transformer to the d input
        v_emb = self.encoder_layer_v(v_input)
        d_emb = self.encoder_layer_d(d_input, src_key_padding_mask=d_input_mask)

        # combine the output of the first 2 transformers and add the dummy word (cls token)
        # put the embedded video and dialog states at the beginning of the combined input
        v_state_emb = v_emb[:, 0, :].unsqueeze(1)
        d_state_emb = d_emb[:, 0, :].unsqueeze(1)
        combined_input = torch.concat([dummy_word, v_state_emb, d_state_emb, v_emb[:, 1:, :], d_emb[:, 1:, :]], axis=1)

        # create combined_input_mask based on the language_emb_mask
        return combined_input