import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class Batch:
    query: torch.Tensor
    query_mask: torch.Tensor
    vft: torch.Tensor
    turns: torch.Tensor
    turns_mask: torch.Tensor
    q_turns: torch.Tensor
    q_turns_mask: torch.Tensor
    a_turns: torch.Tensor
    a_turns_mask: torch.Tensor
    answer: torch.Tensor
    answer_mask: torch.Tensor
    answer_candidates: Optional[torch.Tensor] = None
    answer_candidates_mask: Optional[torch.Tensor] = None


# ---- TODO: Replace with function for the Mask RCNN features ----
def create_monet_like_vft(vft):
    target_dim = 36
    remainder = vft.shape[1] % target_dim
    vft = vft[:, :-remainder].reshape((vft.shape[0], -1, target_dim))
    vft = vft.unsqueeze(3)
    return vft


def batch_interface_simmc2_to_dvd(batch, feature_type):
    if feature_type == 'resnet50':
        vft = batch['features']
        vft = vft.unsqueeze(3)
    elif feature_type == "object_text_features":
        vft = batch['object_features']
        # add frame dimension (only one frame in this cas)
        #vft = vft.unsqueeze(1)
        # bring the data to the format [batch_size x frames x emb_dim (desc_text_len) x obj_number]
        #vft = vft.permute(0, 1, 3, 2)

    batch_in_dvd_format = Batch(
        query=batch['query'],
        query_mask=(batch['query'] != 1),
        vft=vft, 
        turns=batch['turns'], 
        turns_mask=(batch['turns'] != 1), 
        q_turns=batch['q_turns'], 
        q_turns_mask=(batch['q_turns'] != 1),
        a_turns=batch['a_turns'], 
        a_turns_mask=(batch['a_turns'] != 1), 
        answer=batch['answer'].type(torch.int64),
        answer_mask=(batch['answer'] != 1),
        answer_candidates=batch['answer_candidates'],
        answer_candidates_mask=(batch['answer_candidates'] != 1)
    )
    return batch_in_dvd_format



def batch_interface_avsd_to_dvd(batch, feature_type):
    # map question to query
    query = batch['ques'][:,-1, :]
    query_mask = (query != 1)

    # map vid_feat to vft
    # TODO: Use other video features ------!!!-------
    if feature_type == 'i3d':
        vft = create_monet_like_vft(batch['vid_feat'])
    else:
        vft = batch['vid_feat']


    q_turns = batch['ques'][:, :9, :]
    q_turns_mask = (q_turns != 1)

    index_tensor = batch['ans_ind'].unsqueeze(2)
    index_tensor = index_tensor.repeat(1,1,20)
    index_tensor = index_tensor.unsqueeze(2)
    a_turns = batch['opt'].gather(2, index_tensor)
    a_turns = a_turns.squeeze(2)

    # turns should only contain the previous questions (first 9 turns)
    a_turns, answer = a_turns.split([9, 1], dim=1)
    answer = answer.squeeze(1)
    a_turns_mask = (a_turns != 1)
    answer_mask = (answer != 1)

    # concat questions and a_turns to create turns tensor 
    turns = torch.concat((q_turns, a_turns), 2)
    turns_mask = (turns != 1)

    batch_in_dvd_format = Batch(
        query,
        query_mask,
        vft, 
        turns, 
        turns_mask, 
        q_turns, 
        q_turns_mask,
        a_turns,
        a_turns_mask, 
        answer,
        answer_mask
    )
    return batch_in_dvd_format