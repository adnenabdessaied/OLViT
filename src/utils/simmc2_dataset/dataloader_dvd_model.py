#! /usr/bin/env python
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

Dataloader for ambiguous candidates identification task on SIMMC 2.1.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def pad_seq(seqs, pad_token, return_lens=False, is_vft=False):
    lengths = [s.shape[1] for s in seqs]
    max_length = max(lengths)
    output = []
    for seq in seqs:
        if is_vft:
            if len(seq.shape)==4: # spatio-temporal feature
                result = torch.ones(((1, max_length), seq.shape[1], seq.shape[2], seq.shape[3]), dtype=seq.dtype)*pad_token
            else:
                result = torch.ones(((1, max_length), seq.shape[-1]), dtype=seq.dtype)*pad_token
        else:
            result = torch.ones((1, max_length), dtype=seq.dtype)*pad_token
        result[0, :seq.shape[1]] = seq 
        output.append(result)
    if return_lens:
        return lengths, output
    return output 
    

def pad_2d_seq(seqs, pad_token, return_lens=False, is_vft=False):
    lens1 = [len(s) for s in seqs]
    max_len1 = max(lens1)
    all_seqs = []
    for seq in seqs:
        all_seqs.extend(seq)
    lens2 = [s.shape[1] for s in all_seqs]
    max_len2 = max(lens2)
    output = []
    all_lens = []
    for seq in seqs:
        if is_vft:
            result = torch.ones((max_len1, max_len2, seq[0].shape[-1]))*pad_token
        else:
            result = torch.ones((1, max_len1, max_len2))*pad_token
        #turn_lens = torch.ones(max_len1, dtype=np.int)
        offset = max_len1 - len(seq) 
        for turn_idx, turn in enumerate(seq):
            #result[turn_idx,:turn.shape[0]] = turn
            # padding should be at the first turn idxs (Reason: result of last n turns is used for state creation)
            result[0, turn_idx + offset,:turn.shape[1]] = turn
            #turn_lens[turn_idx] = turn.shape[0]
        output.append(result)
    return output


class Simmc2Dataset(Dataset):
    def __init__(self, tokenizer, feature_loader, load_path, args, hidden_labels=False):
        self._tokenizer = tokenizer
        self._features = feature_loader
        self._args = args
        self._hidden_labels = hidden_labels
        print("Loading: {}".format(load_path))
        with open(load_path, "r") as file_id:
            self._raw_data = json.load(file_id)
        # Also read the source data for evaluation.
        with open(self._raw_data["source_path"], "r") as file_id:
            self.source_data = json.load(file_id)
        self._data = self._raw_data["data"]

        self.num_utterances = 2 * args.max_turns + 1
        self.num_instances = len(self._data)
        self.device = torch.cuda if args.use_gpu else torch

    def get_random_batch(self, batch_size):
        indices = np.random.randint(0, self.num_instances, batch_size)
        return self.get_indexed_data(indices)

    def get_entire_batch(self, batch_size):
        all_indices = np.arange(self.num_instances)
        for start in all_indices[::batch_size]:
            batch_indices = all_indices[start : start + batch_size]
            yield self.get_indexed_data(batch_indices)

    
    def __len__(self):
        return len(self._data)

    
    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in ['query', 'answer']:
                seq = pad_seq(merged_batch[key], pad_token=1)
                out[key] = torch.concat(seq, dim=0)
            elif key in ['q_turns', 'a_turns', 'turns', 'object_features', 'answer_candidates']:
                if merged_batch[key][0] is not None:
                    seq = pad_2d_seq(merged_batch[key], pad_token=1)
                    out[key] = torch.concat(seq, dim=0).type(torch.int)
                else:
                    out[key] = None

            elif key in ['features']:
                #features = [f.unsqueeze(1) for f in merged_batch[key]]
                # pad video featues
                features = pad_sequence(merged_batch[key], batch_first=True)
                out[key] = features
            else:
                out[key] = merged_batch[key]
                

        return out

    
    def encode_turns(self, turns):
        encoded_turns = []
        for turn in turns:
            encoded_turn = self._tokenizer(
                turn,
                padding=True,
                max_length=self._args.max_length,
                return_tensors="pt",
                truncation=True,
            )
            encoded_turns.append(encoded_turn['input_ids'].type(torch.int))
        return encoded_turns


    def __getitem__(self, index):
        text_labels = []
        text_inputs = []
        dialog_ids = []
        turn_ids = []
        features = []
        object_maps = []
        # Add <USER> and <SYS> tokens.
        dialog_datum = self._data[index]
        #dialog = self._data[index]["input_text"]
        query = self._data[index]["query"]
        answer = self._data[index]["answer"]
        turns = self._data[index]["turns"]
        q_turns = self._data[index]["q_turns"]
        a_turns = self._data[index]["a_turns"]
        object_features = self._data[index]["object_metadata"]
        if "answer_candidates" in self._data[index].keys():
            answer_candidates = self._data[index]["answer_candidates"]
        else:
            answer_candidates = None

        if self._features:
            feature = self._features[dialog_datum["image_name"]]

        encoded_query = self._tokenizer(
            query,
            padding=True,
            max_length=self._args.max_length,
            return_tensors="pt",
            truncation=True,
        )['input_ids'].type(torch.int)
        encoded_answer = self._tokenizer(
            answer,
            padding=True,
            max_length=self._args.max_length,
            return_tensors="pt",
            truncation=True,
        )['input_ids'].type(torch.int)
        encoded_q_turns = self.encode_turns(q_turns)
        encoded_a_turns = self.encode_turns(a_turns)
        encoded_turns = self.encode_turns(turns)
        encoded_object_features = self.encode_turns(object_features)
        if "answer_candidates" in self._data[index].keys():
            encoded_answer_candidates = self.encode_turns(answer_candidates)
        else:
            encoded_answer_candidates = None


        # Pack the sample.
        sample = {
            "query": encoded_query,
            "answer": encoded_answer,
            "answer_candidates": encoded_answer_candidates,
            "turns": encoded_turns,
            "q_turns": encoded_q_turns,
            "a_turns": encoded_a_turns,
            "object_features": encoded_object_features,
            "dialog_id": dialog_datum["dialog_id"],
            "turn_id": dialog_datum["turn_id"],
            "features": feature,
        }
        return sample


class VisualFeatureLoader:
    """Loads visual features for SIMMC 2.1 ambiguous candidate identification."""

    UNAVAILABLE_IMAGES = [
        "cloth_store_1416238_woman_20_6.png",
        "cloth_store_1416238_woman_19_0.png",
        "cloth_store_1416238_woman_4_8.png",
    ]

    def __init__(self, feature_path, feature_size):
        """Read the features from the path."""
        self._features = torch.load(feature_path)
        self._feature_size = feature_size
        self._zero_feature = torch.zeros((1, self._feature_size), dtype=torch.float)

    def __getitem__(self, label):
        """Get the feature given image label."""
        assert (
            label in self._features or label in self.UNAVAILABLE_IMAGES
        ), f"{label} not found!"
        if label in self.UNAVAILABLE_IMAGES:
            return self._zero_feature
        return self._features[label]

    def cuda(self):
        """Move the features to cuda."""
        self._zero_feature = self._zero_feature.cuda()
        for key, val in self._features.items():
            self._features[key] = val.cuda()
