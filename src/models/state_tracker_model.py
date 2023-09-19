import pytorch_lightning as pl
import torch
from torch import nn
from src.models.base_model import TransformerModel
from src.utils.save_attention_weights import SaveOutput
from src.utils.custom_transformer_encoder_layer import CustomTransformerEncoderLayer
from src.state_trackers.video_state_tracker import VstLSTM
from src.state_trackers.dialogue_state_tracker import DstLSTM
from src.state_trackers.vst_transformer_based import VstTransformer
from src.state_trackers.dst_transformer_based import DstTransformer
from src.combiner.option_a import CombinerOptionA
from src.combiner.option_b import CombinerOptionB
from src.combiner.option_c import CombinerOptionC


class StateTrackerModel(TransformerModel):
    def __init__(self, config, output_path=None):
        super().__init__(config, output_path=output_path)
        self.config = config['model']
        self.ext_config = config['extended_model']

        combine_state_and_emb_options = {
            'OptionA': CombinerOptionA,
            'OptionB': CombinerOptionB,
            'OptionC': CombinerOptionC,
        }
        state_tracker_options = {
            'Transformer': {
                'vst': VstTransformer,
                'dst': DstTransformer
            },
            'LSTM': {
                'vst': VstLSTM,
                'dst': DstLSTM
            }
        }

        # if option b is used the state vector is appended to each embedding -> input size for the transformers needs to double
        if self.ext_config['combiner_option'] == 'OptionB':
            self.model_input_dim *= 2
            # replace fc layer with a fitting one for the larger embeddings
            self.fc = nn.Linear(self.model_input_dim, self.config["fc_dim"])

        self.combiner = combine_state_and_emb_options[self.ext_config['combiner_option']](
            config = self.ext_config,
            model_input_dim = self.model_input_dim,
            use_v_state=self.ext_config['use_v_state'],
            use_d_state=self.ext_config['use_d_state']
        )
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.model_input_dim,
            batch_first=True,
            dropout=self.config['dropout_p'],
            dim_feedforward=self.config['dim_feedforward'],
            nhead=self.config['n_heads']
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config['n_encoder_layers'],
        )
        self.save_output = SaveOutput()
        self.hook_handle = self.encoder.layers[-1].self_attn.register_forward_hook(self.save_output)
        if self.ext_config['use_v_state']:
            self.video_state_tracker = state_tracker_options[self.ext_config['state_tracker_type']]['vst'](
                self.model_input_dim,
                self.config['dropout_p'],
                self.ext_config
            )
        if self.ext_config['use_d_state']:
            self.dial_state_tracker = state_tracker_options[self.ext_config['state_tracker_type']]['dst'](
                self.model_input_dim,
                self.config['dropout_p'],
                self.ext_config
            )
        self.video_emb_start_idx = self.calculate_video_emb_start_idx()


    def calculate_video_emb_start_idx(self):
        video_emb_start_idx = 0
        if self.config['model_type'] == 'discriminative': video_emb_start_idx += 1
        if self.ext_config['use_v_state']: video_emb_start_idx += 1
        if self.ext_config['use_d_state']: video_emb_start_idx += 1
        return video_emb_start_idx


    def determine_relevant_obj_emb(self, attention_weights, vft):
        # determine index of maximum values 
        obj_emb = self.prepare_video_emb(vft)
        _, relevant_emb_indices = attention_weights[:, self.video_emb_start_idx:obj_emb.shape[1] + self.video_emb_start_idx].topk(k=self.ext_config['number_of_relevant_emb'], dim=1)
        relevant_emb = torch.zeros((obj_emb.shape[0], self.ext_config['number_of_relevant_emb'], obj_emb.shape[2]), device=self.device)
        for i in range(attention_weights.shape[0]):
            relevant_emb[i, :, :] = obj_emb[i, relevant_emb_indices[i, :]]

        return relevant_emb


    def get_attention_weights(self, n_vid_emb):
        if self.config['model_type'] in ['generative', 'ranking']:
            # get the attention weights from the query tokens and sum all of them
            query_start_idx = self.video_emb_start_idx + n_vid_emb
            attention_weights = self.save_output.outputs[1][:, query_start_idx:, :]
            attention_weights = attention_weights.sum(dim=1)
        elif self.config['model_type'] == 'discriminative':
            # get only the attention weights of the cls token
            attention_weights = self.save_output.outputs[1][:, 0, :]
        return attention_weights

    
    def forward(self, batch):
        # initialize the state vectors - initialize as none if we dont want to use them
        if self.ext_config['use_v_state']:
            video_state = torch.zeros((batch.query.shape[0], 1, self.model_input_dim), device=self.device)
        else: 
            video_state = None
        if self.ext_config['use_d_state']:
            dial_state = torch.zeros((batch.query.shape[0], 1, self.model_input_dim), device=self.device)
        else:
            dial_state = None

        # create the state vectors based on the previous n most recent dialogue turns
        hist_start_turn_state_gen = batch.turns.shape[1] - 1 - self.ext_config["hist_len_for_state_gen"]
        for dialogue_round in range(max(0, hist_start_turn_state_gen), batch.turns.shape[1]):
            question = batch.q_turns[:, dialogue_round, :]

            question_mask = batch.q_turns_mask[:, dialogue_round, :]
            qa_pair = batch.turns[:, dialogue_round, :]
            qa_pair_mask = batch.turns_mask[:, dialogue_round, :]

            # pass correct answer tokens to the decoder for training a generative model
            if self.config['model_type'] in ['generative', 'ranking']:
                answer = batch.a_turns[:, dialogue_round, :] 
                answer_mask = batch.a_turns_mask[:, dialogue_round, :]
                # the answer is not used, only the attention weights are relevant for state creation
                _ = self.answer_query(question, question_mask, batch.vft, video_state, dial_state, answer, answer_mask, state_generation_mode=True)
            else:
                _ = self.answer_query(question, question_mask, batch.vft, video_state, dial_state)


            # update the states
            if self.ext_config['use_v_state']:
                # get the attention weights from the last "answer_query" call and determine the relevant obj
                attention_weights = self.get_attention_weights(n_vid_emb=batch.vft.shape[1])
                relevant_obj_emb = self.determine_relevant_obj_emb(attention_weights, batch.vft)
                 # add ids to match the input size of the main transformer block
                video_state = self.video_state_tracker(relevant_obj_emb)
            if self.ext_config['use_d_state']:
                qa_pair_emb = self.prepare_lang_emb(qa_pair, qa_pair_mask)
                # add ids to match the input size of the main transformer block
                dial_state = self.dial_state_tracker(qa_pair_emb)
        
        # delete state of the state tracker
        if self.ext_config['use_v_state']:
            self.video_state_tracker.reset()
        if self.ext_config['use_d_state']:
            self.dial_state_tracker.reset()

        # answer the actual question
        # pass correct answer tokens to the decoder for training a generative model
        if self.config['model_type'] in ['generative', 'ranking']:
            output = self.answer_query(batch.query, batch.query_mask, batch.vft, video_state, dial_state, batch.answer, batch.answer_mask)
        else:
            output = self.answer_query(batch.query, batch.query_mask, batch.vft, video_state, dial_state)
          
        return output


        
