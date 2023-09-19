# code is partly inspired from https://pytorch.org/tutorials/beginner/translation_transformer.html

from unittest import result
from src.models.state_tracker_model import StateTrackerModel
from src.utils.batch_interfaces import batch_interface_simmc2_to_dvd, batch_interface_avsd_to_dvd
from dataclasses import dataclass
import torch
from torch import nn
from torchtext.data.metrics import bleu_score
import json
import os
from transformers import AutoTokenizer
import nltk
import numpy as np
from src.utils.text_utils import normalize_sentence, translate_from_ids_to_text  




class GenerativeModel(StateTrackerModel):
    def __init__(self, config, output_path=None):
        super().__init__(config, output_path=output_path)

        self.transformer = nn.Transformer(
            d_model=self.model_input_dim,
            batch_first=True,
            dropout=self.config['dropout_p'],
            dim_feedforward=self.config['dim_feedforward'],
            nhead=self.config['n_heads'],
            num_encoder_layers=self.config['n_encoder_layers'],
            num_decoder_layers=self.config['n_decoder_layers'],
            custom_encoder=self.encoder
        )
        self.prob_generator = nn.Linear(self.model_input_dim, self.config['vocab_size'])

        self.pad_id = 1
        self.unk_id = 3
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_id)


        # tokenizer for translation from ids to text 
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['pretrained_lm_name'])

        # ---TODO: Remove ------
        self.results = {} 
        self.epoch_count = 0


        # -----------------------
        self.batch_interface = batch_interface_simmc2_to_dvd


    def encode_object_descriptions(self, vft):
        #embed the object descriptions using bert and then create the object token using transformer layers
        if self.config['feature_type'] == "object_text_features":
            object_features = []
            for i in range(vft.shape[1]):
                object_description = vft[:, i, :]
                object_description_mask = (object_description != 1)
                embedded_object_description = self.apply_pretrained_lm(object_description, object_description_mask)
            
                #map embeddings to a smaller size (motivation: reduce transformer sice of object description encoder)
                embedded_object_description = self.linear_projection_object_description(embedded_object_description)

                #apply transformer to encode the object description
                object_token = self.object_description_encoder(embedded_object_description)
                object_features.append(object_token)
            object_features = torch.concat(object_features, dim=1)
            #add frame dimension (only one frame in this cas)
            object_features = object_features.unsqueeze(1)
            #bring the data to the format [batch_size x frames x emb_dim (desc_text_len) x obj_number]
            vft = object_features.permute(0, 1, 3, 2)

        return vft


    def create_target_mask(self, size):
        mask = torch.triu(torch.ones((size,size), device=self.device), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))   
        return mask

    
    def generate_prob_for_next_tokens(self, input, answer_emb, tgt_mask, input_mask, answer_mask):
        x = self.transformer.encoder(input, src_key_padding_mask=input_mask)
        dec_out = self.transformer.decoder(answer_emb, x, tgt_mask)
        probs = self.prob_generator(dec_out)


        return probs


    def generate_complete_answers(self, input, input_mask):
        # encode the complete batch of questions
        memory = self.transformer.encoder(input, src_key_padding_mask=input_mask)
        generated_answers = torch.ones(memory.shape[0], 40, dtype=torch.int) # 20 = max answer length, use unknown token ()

        # generate the answers for each individual question from the batch
        for i in range(memory.shape[0]):
            memory_i = memory[i, :, :]
            memory_i = memory_i.unsqueeze(0)
            answer_i = torch.zeros((1,1), dtype=torch.int, device=self.device) # Pass start token <s> to decoder as first input. From roberta vocab: <s>": 0, "</s>": 2
             
            for j in range(40): # 20 = max answer length

                answer_i_emb = self.prepare_lang_emb(answer_i, torch.ones((1, answer_i.shape[0]), device=self.device, dtype=torch.int16))             
                tgt_mask = self.create_target_mask(answer_i.shape[1])          
                decoder_output = self.transformer.decoder(answer_i_emb, memory_i, tgt_mask)
                prob = self.prob_generator(decoder_output[:, -1, :])
                next_word = prob.argmax()

                answer_i = torch.concat([answer_i, next_word.unsqueeze(0).unsqueeze(0)], dim=1)
                if next_word.item() == 2: # eos token in roberta vocab "</s>": 2
                    break
            
            generated_answers[i, :answer_i.shape[1] - 1] = answer_i[0, 1:]
        
        return generated_answers


    def apply_model(self, language_emb, language_emb_mask, video_emb, v_state=None, d_state=None, answer_emb=None, answer_mask=None, state_generation_mode=False):
        # combine state and embeddings
        input = self.combiner(
                video_emb,
                language_emb,
                language_emb_mask,
                v_state,
                d_state
        )
        # create input mask based on the language_emb_mask (complete video is unmasked)
        input_mask = torch.zeros((input.shape[0], input.shape[1]), device=self.device)
        offset = 0
        if v_state is not None: offset += 1 
        if d_state is not None: offset += 1 
        # offset is caused by state vectors
        input_mask[:, video_emb.shape[1] + offset:] = ~language_emb_mask
        tgt_mask = self.create_target_mask(answer_emb.shape[1])

        #-------TODO: Mask padded object embeddings when text based object embeddings are used -------------

        if self.mode == 'train' or state_generation_mode:
            probs = self.generate_prob_for_next_tokens(input, answer_emb, tgt_mask, input_mask, answer_mask)
            return probs
        elif self.mode == 'val':
            generated_answers = self.generate_complete_answers(input, input_mask)
            return generated_answers


    def prepare_answer_emb_and_mask(self, answer, answer_mask):
        mask = torch.tril(torch.ones((answer.shape[1], answer.shape[1]), device=self.device))
        mask = mask.unsqueeze(0)
        mask = mask.expand(answer.shape[0], -1, -1)
        answer_emb = self.apply_pretrained_lm(answer, mask)

        answer_emb = self.linear_projection_text(answer_emb)
        answer_emb = self.append_ids(answer_emb, [1, 0], 2)
        answer_emb = self.positional_encoder(answer_emb)

        # pytorch interprets True in a mask as padding 
        answer_mask = ~answer_mask
        answer_emb_final = answer_emb[:, :-1].detach()
        answer_mask_final = answer_mask[:, :-1].detach()

        return answer_emb_final, answer_mask_final


    def answer_query(self, query, query_mask, vft, v_state=None, d_state=None, answer=None, answer_mask=None, state_generation_mode=False):
        video_emb = self.prepare_video_emb(vft)
        lang_emb = self.prepare_lang_emb(query, query_mask)
        answer_emb, answer_mask = self.prepare_answer_emb_and_mask(answer, answer_mask)
        output = self.apply_model(lang_emb, query_mask, video_emb, v_state, d_state, answer_emb, answer_mask, state_generation_mode)
        return output

    
    def training_step(self, train_batch, batch_idx):
        train_batch = self.batch_interface(train_batch, feature_type=self.config['feature_type'])
        if self.config['feature_type'] == "object_text_features":
            train_batch.vft = self.encode_object_descriptions(train_batch.vft)

        logits = self.forward(train_batch)
        logits = logits.permute(0, 2, 1)

        # replace any unknown token (id = 3) with a padding token in order to also ignore them -> avoid model which outputs unk tokens
        train_batch.answer[train_batch.answer == 3] = 1
        loss = self.loss(logits, train_batch.answer[:, 1:]) # ignore padding 
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=train_batch.query.shape[0])
        return loss


    def get_next_token_pred_as_text_and_logits(self, batch):
        # set mode to train to get the logits instead of completely generated sentences
        self.mode = 'train'
        logits = self.forward(batch)
        logits = logits.permute(0, 2, 1)
        predicted_tokens = []
        for j in range(logits.shape[0]):
            l = logits[j, :, :]
            ids = [l[:, i].argmax().item() for i in range(l.shape[1])]
            text = translate_from_ids_to_text(ids, self.tokenizer)
            predicted_tokens.append(text)
        # set mode back to val 
        self.mode = 'val'
        
        return predicted_tokens, logits

    
    def calculate_bleu_score(self, generated_answer_ids, correct_answer):
        # calculate bleu score for the generated answers compared to the provided correct answers
        bleu4_scores = []
        all_generated_answers = []
        for i in range(generated_answer_ids.shape[0]):
            generated_answer = generated_answer_ids[i, :].tolist()
            generated_answer_text = translate_from_ids_to_text(generated_answer, self.tokenizer)
            all_generated_answers.append(generated_answer_text)
            correct_answer_text_i = correct_answer[i]
            score4 = nltk.translate.bleu_score.sentence_bleu(
                [normalize_sentence(correct_answer_text_i)],
                normalize_sentence(generated_answer_text),
                smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7    
            )
            bleu4_scores.append(score4)
        bleu4_score = np.mean(bleu4_scores) 
        return bleu4_score, all_generated_answers


    def translate_answer_ids_to_text(self, answer):
        correct_answer_text = []
        for i in range(answer.shape[0]):
            correct_answer_i = answer[i, :].tolist()
            correct_answer_text_i = translate_from_ids_to_text(correct_answer_i, self.tokenizer)
            correct_answer_text.append(correct_answer_text_i)
        return correct_answer_text


    def validation_step(self, val_batch, batch_idx):
        val_batch = self.batch_interface(val_batch, feature_type=self.config['feature_type'])
        if self.config['feature_type'] == "object_text_features":
            val_batch.vft = self.encode_object_descriptions(val_batch.vft)

        correct_answer_text = self.translate_answer_ids_to_text(val_batch.answer)
        generated_answer_ids = self.forward(val_batch)

        # calculate and log bleu score for the generated answers compared to the provided correct answers
        bleu4_score, generated_answers_text = self.calculate_bleu_score(generated_answer_ids, correct_answer_text)
        self.log('bleu4', bleu4_score, prog_bar=True, on_step=False, on_epoch=True, batch_size=generated_answer_ids.shape[0])    

        # calculate and log the validation loss based on the results from next token predicition (train mode needed)        
        predicted_tokens, logits = self.get_next_token_pred_as_text_and_logits(val_batch)        
        loss = self.loss(logits, val_batch.answer[:, 1:]) # ignore padding 
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=val_batch.query.shape[0])

        return {'next_token_predictions': predicted_tokens, 'generated_answers': generated_answers_text, 'correct_answers': correct_answer_text}


    def test_step(self, test_batch, batch_idx):
        dialog_id = test_batch['dialog_id']
        turn_id = test_batch['turn_id']
        test_batch = self.batch_interface(test_batch, feature_type=self.config['feature_type'])
        if self.config['feature_type'] == "object_text_features":
            test_batch.vft = self.encode_object_descriptions(test_batch.vft)

        correct_answer_text = self.translate_answer_ids_to_text(test_batch.answer)
        generated_answer_ids = self.forward(test_batch)

        # calculate and log bleu score for the generated answers compared to the provided correct answers
        bleu4_score, generated_answers_text = self.calculate_bleu_score(generated_answer_ids, correct_answer_text)
        self.log('bleu4', bleu4_score, prog_bar=True, on_step=False, on_epoch=True, batch_size=generated_answer_ids.shape[0])    

        # calculate and log the validation loss based on the results from next token predicition (train mode needed)        
        predicted_tokens, logits = self.get_next_token_pred_as_text_and_logits(test_batch)        
        loss = self.loss(logits, test_batch.answer[:, 1:]) # ignore padding 
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=test_batch.query.shape[0])

        return {'turn_id': turn_id, 'next_token_predictions': predicted_tokens, 'dialog_id': dialog_id, 'generated_answers': generated_answers_text, 'correct_answers': correct_answer_text}
    

    def test_epoch_end(self, outputs):

        if self.config['output_format'] == 'submission':
            responses = []
            for output in outputs:
                for t_id, d_id, answer in zip(output['turn_id'], output['dialog_id'], output['generated_answers']):
                    sample = {
                        'dialog_id': d_id,
                        'predictions': [
                            {
                                'turn_id': t_id,
                                'response': answer
                            }
                        ]
                    }
                    responses.append(sample)
            name = 'dstc11-simmc-devtest-pred-subtask-4-generation.json'
            with open(os.path.join(self.output_path, name), 'w') as file:
                json.dump(responses, file)        

        else:
            result_idx = 0
            for output in outputs:
                for j in range(len(output['next_token_predictions'])):
                    pred = " "
                    corr = " "
                    gen = " "
                    self.results[result_idx] = {
                        'next_token_pred': pred.join(output['next_token_predictions'][j]),
                        'generated_ans': gen.join(output['generated_answers'][j]),
                        'correct': corr.join(output['correct_answers'][j])
                    }
                    result_idx += 1

            name = f'epoch_{self.epoch_count}.json'
            with open(os.path.join(self.output_path, name), 'w') as file:
                json.dump(self.results, file)

    
    def validation_epoch_end(self, outputs):
        result_idx = 0
        for output in outputs:
            for j in range(len(output['next_token_predictions'])):
                pred = " "
                corr = " "
                gen = " "
                self.results[result_idx] = {
                    'next_token_pred': pred.join(output['next_token_predictions'][j]),
                    'generated_ans': gen.join(output['generated_answers'][j]),
                    'correct': corr.join(output['correct_answers'][j])
                }
                result_idx += 1

        name = f'epoch_{self.epoch_count}.json'
        with open(os.path.join(self.output_path, name), 'w') as file:
            json.dump(self.results, file)

        self.results = {}
        self.epoch_count += 1


    def on_train_epoch_start(self):
        self.mode = 'train' 


    def on_validation_epoch_start(self):
        self.mode = 'val'

    
    def on_test_epoch_start(self):
        self.mode = 'val'

    


