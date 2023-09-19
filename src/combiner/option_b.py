import pytorch_lightning as pl
import torch

class CombinerOptionB(pl.LightningModule):
    def __init__(self, config=None, model_input_dim=None, use_v_state=False, use_d_state=False):
        super().__init__()
        self.use_v_state = use_v_state
        self.use_d_state = use_d_state

    
    def append_state_to_emb(self, tensor, state):
        tiling_vector = [1, tensor.shape[1], 1]
        state_tensor_for_concatenation = torch.tile(state, tiling_vector)
        result = torch.concat([tensor, state_tensor_for_concatenation], axis=2)
        return result 


    def forward(self, dummy_word, video_emb, language_emb, language_emb_mask, v_state, d_state):
        # concatenate the video emb with the video state and the language emb with the dialogue state
        # if the stat is not used, concatenate itself   
        if v_state is not None \
             and d_state is not None \
             and self.use_v_state \
             and self.use_d_state:
            video_emb = self.append_state_to_emb(video_emb, v_state)
            language_emb = self.append_state_to_emb(language_emb, d_state)
        elif d_state is not None and self.use_d_state:
            video_emb = self.append_state_to_emb(video_emb, video_emb)
            language_emb = self.append_state_to_emb(language_emb, d_state)
        elif v_state is not None and self.use_v_state:
            video_emb = self.append_state_to_emb(video_emb, v_state)
            language_emb = self.append_state_to_emb(language_emb, language_emb)
        else:
            video_emb = self.append_state_to_emb(video_emb, video_emb)
            language_emb = self.append_state_to_emb(language_emb, language_emb)

        output = torch.concat([dummy_word, video_emb, language_emb], axis=1)
        return output
