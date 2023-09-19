"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy, logging, sys, time, os, pdb, random, glob, json
import pickle as pkl
import numpy as np
from tqdm import tqdm
from collections import Counter
from functools import partial
import nltk
import torch 
import torch.utils.data as Data
from src.utils.dvd_codebase.data.dataset import *
from src.utils.dvd_codebase.data.analysis_utils import *
from src.utils.dvd_codebase.data.data_utils import *
from src.utils.dvd_codebase.data.analysis_utils import get_question_subtype, get_question_complexity
from transformers import AutoTokenizer


def load_dials(args, split):
    files = []
    for video_split in ['all_actions', 'max2action']:
        files += glob.glob(args.data_dir + '{}_{}_*/*.json'.format(video_split, split))
    files = sorted(files)  # [:50]
    if args.debug:
        files = files[:100]
    all_dials = []
    vid_set = {}
    for file in tqdm(files, total=len(files)):
        dials = json.load(open(file))
        all_dials.extend(dials)
        video_split = dials[0][0]['split']
        vid = dials[0][0]['image'].replace('CLEVR', 'CATER')
        vid_key = '{}-{}'.format(video_split, vid)
        if vid_key not in vid_set:
            vid_set[vid_key] = '{}/{}/{}.pkl'.format(args.fea_dir, video_split, vid)
    return all_dials, vid_set

def load_videos(args, vid_set):
    vid_fts = {}
    ft_dims = None
    size, stride = -1, -1
    segment_map = {}
    for vid_key, fea_file in tqdm(vid_set.items(), total=len(vid_set)):
        #fea_file = '{}/{}.pkl'.format(args.fea_dir, vid)        
        fea = pkl.load(open(fea_file, 'rb'))
        output = []
        for clip_idx, clip in enumerate(fea['clips']): 
            fea = clip['features']
            if len(fea.shape)==3:
                fea = fea.transpose(1, 2, 0)
            output.append(fea)
            start, end = clip['segment']
            if clip_idx not in segment_map:
                segment_map[clip_idx] = (start, end)
            if size == -1:
                size = end - start + 1
            if clip_idx>0 and stride == -1:
                stride = start - prior_start
            prior_start, prior_end = start, end 
        vft = np.asarray(output)
        vid_fts[vid_key] = vft 
        if ft_dims is None:
            ft_dims = vft.shape
    return vid_fts, ft_dims, size, stride, segment_map

def load_video_features(args, vid_set):
    vid_fts = {}
    for vid_key, fea_file in tqdm(vid_set.items(), total=len(vid_set)):
        #fea_file = '{}/{}.pkl'.format(args.fea_dir, vid)        
        fea = pkl.load(open(fea_file, 'rb'))
        vid_fts[vid_key] = fea
    return vid_fts
    
def get_vocabulary(dials, args, vocab=None):
    #answer_options = set()
    word_freq = {}
    for dialog in tqdm(dials, total=len(dials)):
        for turn in dialog:
            for word in nltk.word_tokenize(turn['question']):
                if word not in word_freq: word_freq[word] = 0
                word_freq[word] += 1                    
            answer = str(turn['answer'])
            #answer_options.add(answer)
            for word in nltk.word_tokenize(answer):
                if word not in word_freq: word_freq[word] = 0
                word_freq[word] += 1 
            program = turn['final_all_program']
            for n in program: 
                if n['type'] == 'identity': continue 
                if n['type'] not in word_freq: word_freq[n['type']] = 0
                word_freq[n['type']] += 1     
                if 'side_inputs' in n:
                    for side_input in n['side_inputs']:
                        for word in nltk.word_tokenize(side_input):
                            if word not in word_freq: word_freq[word] = 0
                            word_freq[word] += 1                           
    if vocab is not None: 
        unk_words = set()
        for word, freq in word_freq.items():
            if word not in vocab:
                unk_words.add(word)
        return unk_words 
    vocab = {'<unk>':0, '<blank>':1, '<sos>':2, '<eos>':3, '<eoo>': 4}
    for word, freq in word_freq.items():
        vocab[word] = len(vocab) 
    answer_options =  ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'False', 'True', 'blue', 'brown', 'cone', 'cube', 'cyan', 'cylinder', 'flying', 'flying,rotating', 'flying,rotating,sliding', 'flying,sliding', 'gold', 'gray', 'green', 'large', 'medium', 'metal', 'no action', 'purple', 'red', 'rotating', 'rotating,sliding', 'rubber', 'sliding', 'small', 'sphere', 'spl', 'yellow']
    return vocab, answer_options 

def answer_by_question_type(dials):
    qa_dist = {} 
    for dialog in dials:
        for turn_idx, turn in enumerate(dialog):
            answer = turn['answer']
            template = turn['template']
            if turn_idx > 0: 
                prior_template = dialog[turn_idx-1]['template']
            else:
                prior_template = None 
            qtype = get_question_subtype(template, prior_template)
            if qtype not in qa_dist:
                qa_dist[qtype] = {}
            if answer not in qa_dist[qtype]:
                qa_dist[qtype][answer] = 0
            qa_dist[qtype][answer] += 1 
    return qa_dist


# Load text data
def create_dials(dials, vocab, answer_list, vft_data, args, tokenizer=None):
    dialog_list = []
    qa_id = 0
    for dialog in tqdm(dials, total=len(dials)):
        if tokenizer is None:
            questions = [words2ids(t['question'], vocab) for t in dialog]
            answers = [words2ids(str(t['answer']), vocab) for t in dialog]
        else:
            questions = [words2ids_pretrained_lm(t['question'], vocab, tokenizer) for t in dialog]
            answers = [words2ids_pretrained_lm(str(t['answer']), vocab, tokenizer) for t in dialog]
        answer_output = [[answer_list.index(str(t['answer']))] for t in dialog]
        qa_pair = [np.concatenate((q,a)).astype(np.int32) for q,a in zip(questions, answers)]
        
        attribute_dependencies = []
        object_dependencies = []
        temporal_dependencies = []
        spatial_dependencies = []
        q_types = []
        q_complexities = []
        for i, t in enumerate(dialog):
            # determine the type of turn relation
            attribute_dependencies.append(t['turn_dependencies']['attribute'])
            object_dependencies.append(t['turn_dependencies']['object'])
            temporal_dependencies.append(t['turn_dependencies']['temporal'])
            spatial_dependencies.append(t['turn_dependencies']['spatial'])

            # determine the question type based on the template for analysis reasons 
            if i == 0:
                q_types.append(get_question_type(t['template'], None))
            else:
                q_types.append(get_question_type(t['template'], dialog[i-1]['template']))
        
            # get question complexity
            q_complexities.append(get_question_complexity(t, t['template_filename'] ))

        # get image name
        video_name = t['image']           

        vid_cutoffs = [t['template']['cutoff'] for t in dialog]
        gt_vid_periods = [t['template']['used_periods'][-1] for t in dialog]
        programs = [program2ids(t['final_all_program'], vocab) for t in dialog]
        states = [state2ids(t['template']['used_objects'], vocab) for t in dialog]
        vid = dialog[0]['image'].replace('CLEVR', 'CATER')
        vid_split = dialog[0]['split']
        vid_key = '{}-{}'.format(vid_split, vid)
        whole_vft_fea = vft_data[vid_key]
        turn_based_vft_fea = []

        # cutoff the unused vft data based on the vid_cutoffs
        for t_idx, t_cutoff in enumerate(vid_cutoffs):
            if t_cutoff is not None:
                t_vft_fea = whole_vft_fea[:t_cutoff[3], :, :]
            else:
                t_vft_fea = whole_vft_fea
            turn_based_vft_fea.append(t_vft_fea)
        
        for n in range(len(questions)): 
            start_turn_idx = 0 
            history = np.asarray([])
            turns = []
            q_turns = []
            a_turns = []
            for m in range(start_turn_idx, n):
                history = np.append(history, qa_pair[m])
                turns.append(qa_pair[m])
                q_turns.append(questions[m])
                a_turns.append(np.array(answer_output[m]))   
                
            question = questions[n]
            answer = answer_output[n]
            program = programs[n]
            state = states[n]
            gt_period = gt_vid_periods[n]
            q_type = q_types[n]
            attribute_dependency = attribute_dependencies[n]
            object_dependency = object_dependencies[n]
            temporal_dependency = temporal_dependencies[n]
            spatial_dependency = spatial_dependencies[n]
            q_complexity = q_complexities[n]
            vft_feat = turn_based_vft_fea[n]

            item = [vid_split, vid, qa_id, history, question, answer, turns, 
                    q_turns, a_turns, vft_feat, gt_period,
                    program, state, q_type, attribute_dependency, object_dependency,
                    temporal_dependency, spatial_dependency, video_name, q_complexity]

            dialog_list.append(item)
            qa_id += 1
    
    data = {'dialogs': dialog_list, 'vocab': vocab, 'answer': answer_list, 'features': []}
    return data 


def create_dataset(data, vocab, split, args):
    out = {}
    keys = ['vid_split', 'vid', 'qa_id', 'history', 'question', 'answer', 'turns', 
            'q_turns', 'a_turns', 'vft', 'gt_period', 
            'program', 'state', 'q_type', 'attribute_dependency', 'object_dependency',
            'temporal_dependency', 'spatial_dependency', 'video_name', 'q_complexity']
    for key in keys:
        out[key] = []
    for dialog in data['dialogs']:
        out['vid_split'].append(dialog[0])
        out['vid'].append(dialog[1])
        out['qa_id'].append(dialog[2])
        out['history'].append(dialog[3])
        out['question'].append(dialog[4])
        out['answer'].append(dialog[5])
        out['turns'].append(dialog[6])
        out['q_turns'].append(dialog[7])
        out['a_turns'].append(dialog[8])
        out['vft'].append(dialog[9])
        out['gt_period'].append(dialog[10])
        out['program'].append(dialog[11])
        out['state'].append(dialog[12])
        out['q_type'].append(dialog[13])
        out['attribute_dependency'].append(dialog[14])
        out['object_dependency'].append(dialog[15])
        out['temporal_dependency'].append(dialog[16])
        out['spatial_dependency'].append(dialog[17])
        out['video_name'].append(dialog[18])
        out['q_complexity'].append(dialog[19])

    dataset = Dataset(out)         
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=(split=='train'),
                                                  collate_fn=partial(collate_fn, vocab=vocab), 
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
    return data_loader, len(out['vid'])
