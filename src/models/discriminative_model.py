from src.models.state_tracker_model import StateTrackerModel
import torch
from torch import nn
from src.utils.text_utils import translate_from_ids_to_text    
import pandas as pd   


class DiscriminativeModel(StateTrackerModel):
    def __init__(self, config, output_path=None):
        super().__init__(config, output_path=output_path)

        self.fc = nn.Linear(self.model_input_dim, self.config["fc_dim"])
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.config["fc_dim"], 40)


    def apply_model(self, language_emb, language_emb_mask, video_emb, v_state=None, d_state=None, answer_emb=None, answer_mask=None, state_generation_mode=None):
        # analogous to the CLS token from BERT models 
        dummy_word = torch.zeros(self.model_input_dim, requires_grad=True, device=self.device)
        dummy_word = torch.tile(dummy_word, (language_emb.shape[0], 1, 1))

        # combine state and embeddings
        input = self.combiner(
                video_emb,
                language_emb,
                language_emb_mask,
                v_state,
                d_state,
                dummy_word
        )
        # create input mask based on the language_emb_mask (complete video is unmasked)
        input_mask = torch.zeros((input.shape[0], input.shape[1]), device=self.device)
        offset = 1
        if v_state is not None: offset += 1 
        if d_state is not None: offset += 1 
        # offset is caused by cls token and state vectors
        if self.config['model_type'] == 'extended_model':
            # set offset to 1 if combiner B is used -> no state vectors as input. Instead concatenated with embeddings
            if self.ext_config['combiner_option'] == 'OptionB':
                offset = 1
        input_mask[:, video_emb.shape[1] + offset:] = ~language_emb_mask

        x = self.encoder(input, src_key_padding_mask=input_mask)
        # only pass transformed dummy word to the dense layers
        x = self.fc(x[:, 0, :])
        x = self.relu(x)
        output = self.output(x)
        return output


    def answer_query(self, query, query_mask, vft, v_state=None, d_state=None, answer=None, answer_mask=None, state_generation_mode=False):
        video_emb = self.prepare_video_emb(vft)
        lang_emb = self.prepare_lang_emb(query, query_mask)
        if answer is not None and answer_mask is not None:
            answer_emb = self.prepare_lang_emb(answer, answer_mask)
        else:
            answer_emb = None
        output = self.apply_model(lang_emb, query_mask, video_emb, v_state, d_state, answer_emb, answer_mask, state_generation_mode)
        return output


    def training_step(self, train_batch, batch_idx):
        train_batch.move_to_cuda()
        label = torch.squeeze(train_batch.answer)
        out = self.forward(train_batch)
        loss = self.loss(out, label)
        tr_acc = self.train_acc(out.softmax(dim=1), label)
        if tr_acc > self.best_train_acc:
            self.best_train_acc = tr_acc
        self.log("train_acc", tr_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=train_batch.query.shape[0])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=train_batch.query.shape[0])
        print('train_loss: {} | train_acc: {}'.format(loss, tr_acc))
        return loss


    def validation_step(self, val_batch, batch_idx):
        val_batch.move_to_cuda()
        label = torch.squeeze(val_batch.answer)
        out = self.forward(val_batch)
        loss = self.loss(out, label)
        self.val_acc(out.softmax(dim=1), label)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=val_batch.query.shape[0])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=val_batch.query.shape[0])
        return {'val_loss': loss, 'val_acc': self.val_acc.compute()}


    def test_step(self, test_batch, batch_idx):
        test_batch.move_to_cuda()
        label = torch.squeeze(test_batch.answer)
        out = self.forward(test_batch)
        loss = self.loss(out, label)
        self.test_acc(out.softmax(dim=1), label)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=test_batch.query.shape[0])
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=test_batch.query.shape[0])
    
        # save the results into a dictionary
        out = torch.argmax(out, dim=1)

        question_as_text = []
        for i in range(test_batch.query.shape[0]):
            question_ids = test_batch.query[i, :]
            question_as_text.append(translate_from_ids_to_text(question_ids, self.tokenizer))
            
        self.results['question'].extend(question_as_text)
        self.results['video_name'].extend(test_batch.video_name)

        self.results['qa_id'].extend(test_batch.qa_ids)
        self.results['q_type'].extend(test_batch.q_type)
        self.results['label'].extend(label.tolist())
        self.results['output'].extend(out.tolist())
        self.results['attribute_dependency'].extend(test_batch.attribute_dependency)
        self.results['object_dependency'].extend(test_batch.object_dependency)
        self.results['temporal_dependency'].extend(test_batch.temporal_dependency)
        self.results['spatial_dependency'].extend(test_batch.spatial_dependency)
        self.results['q_complexity'].extend(test_batch.q_complexity)


    def on_test_start(self):
        self.results = {
            'qa_id': [],
            'q_type': [],
            'label': [],
            'output': [],
            'attribute_dependency': [],
            'object_dependency': [],
            'temporal_dependency': [],
            'spatial_dependency': [],
            'q_complexity': [],
            # only needed for input output analysis
            'question': [],
            'video_name': []
        }


    def on_test_end(self):
       df = pd.DataFrame.from_dict(self.results)
       df.to_pickle(self.output_path)
