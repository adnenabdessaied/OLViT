import pytorch_lightning as pl
import torch

class CombinerOptionA(pl.LightningModule):
    def __init__(self, config=None, model_input_dim=None, use_v_state=False, use_d_state=False):
        super().__init__()
        self.use_v_state = use_v_state
        self.use_d_state = use_d_state

    def forward(self, vision_emb, language_emb, language_emb_mask, v_state, d_state, dummy_word=None):
        if v_state is not None \
             and d_state is not None \
             and self.use_v_state \
             and self.use_d_state:
            output = torch.concat([v_state, d_state, vision_emb, language_emb], axis=1)
        elif d_state is not None and self.use_d_state:
            output = torch.concat([d_state, vision_emb, language_emb], axis=1)
        elif v_state is not None and self.use_v_state:
            output = torch.concat([v_state, vision_emb, language_emb], axis=1)
        else:
            output = torch.concat([vision_emb, language_emb], axis=1)
        if dummy_word is not None:
            output = torch.concat([dummy_word, output], axis=1)

        return output
