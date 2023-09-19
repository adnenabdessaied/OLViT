import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from src.utils.positional_encoding import PositionalEncoding
from src.object_description_encoder.object_description_encoder import ObjectDescriptionEncoder
import torchmetrics as metrics
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModel
from src.combiner.option_a import CombinerOptionA
from transformers import AutoTokenizer


class TransformerModel(pl.LightningModule):
    def __init__(self, config, output_path=None):
        super().__init__()
        self.output_path = output_path
        self.config = config['model']
        self.train_config = config['training']

        self.train_acc = metrics.Accuracy('multiclass', num_classes=40)
        self.val_acc = metrics.Accuracy('multiclass', num_classes=40)
        self.test_acc = metrics.Accuracy('multiclass', num_classes=40)

        self.best_val_acc = 0
        self.loss_for_best_val_acc = 0
        self.best_train_acc = 0


        self.combiner = CombinerOptionA()
        self.initialize_text_encoder_and_feature_mapping()

        self.positional_encoder = PositionalEncoding(
            d_model=self.model_input_dim, dropout=self.config['dropout_p'], max_len=self.config['dim_feedforward']
        )

        encoder_layer = nn.TransformerEncoderLayer(
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

        self.loss = nn.CrossEntropyLoss()

        if self.config['feature_type'] == 'object_text_features':
            self.object_description_encoder = ObjectDescriptionEncoder(
                d_model=self.config['v_emb_dim'],
                config=self.config
            )
            # maps the output from the pretrained lm to as smaller size used for the encoding of the object description (reduces transformer size)
            self.linear_projection_object_description = nn.Linear(self.pretrained_lm.config.hidden_size, self.config['v_emb_dim'])


        # tokenizer for translation from ids to text 
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['pretrained_lm_name'])


    def initialize_text_encoder_and_feature_mapping(self):
        if self.config['use_pretrained_lm']:
            self.pretrained_lm = AutoModel.from_pretrained(
                self.config['pretrained_lm_name'],
                add_pooling_layer=False
            )
            self.pretrained_lm.eval()
            # don't train the paramteres of the pretrained lm
            self.pretrained_lm.config.training = True
            # for param in self.pretrained_lm.parameters():
            #     param.requires_grad = False

            # initialize the projection layers to map the embeddings to the correct input dim
            # either use the emb_dim as done in aloe (v_emb_dim * n_heads) or the emb_dim specified in the config 
            if self.config['projection_as_in_aloe']: 
                self.model_input_dim = self.config['n_heads'] * self.config['v_emb_dim']
                self.linear_projection_video = nn.Linear(self.config['v_emb_dim'], self.model_input_dim - 2)
                self.linear_projection_text = nn.Linear(self.pretrained_lm.config.hidden_size, self.model_input_dim - 2)
            else:
                # take embedding size from config and map the video features from their size to the chose emb size 
                self.linear_projection_video = nn.Linear(self.config['v_emb_dim'], self.config['emb_dim'] - 2)
                self.linear_projection_text = nn.Linear(self.pretrained_lm.config.hidden_size, self.config['emb_dim'] - 2)
                self.model_input_dim = self.config['emb_dim']
        else:
            # either use the emb_dim as done in aloe (v_emb_dim * n_heads) or the video_emb_dim (2 is either added or subtracted because of the input ids)
            if self.config['projection_as_in_aloe']: 
                self.model_input_dim = self.config['n_heads'] * self.config['v_emb_dim']
            else:
                self.model_input_dim = self.config['emb_dim']
            self.linear_projection_video = nn.Linear(self.config['v_emb_dim'], self.model_input_dim - 2)
            self.embed = nn.Embedding(num_embeddings=self.config['vocab_size'], embedding_dim=self.model_input_dim - 2)


    def append_ids(self, tensor, id_vector, axis):
        id_vector = torch.tensor(id_vector, device=self.device)
        for a in range(len(tensor.shape)):
            if a != axis:
                id_vector = torch.unsqueeze(id_vector, axis=a)
        tiling_vector = [s if i != axis else 1 for i, s in enumerate(tensor.shape)]
        id_tensor = torch.tile(id_vector, tiling_vector)
        return torch.concat([tensor, id_tensor], axis=axis)
    

    def downsample_video_emb(self, video_emb):
        return video_emb[:, ::self.config['sample_rate_video'], :, :]


    def unroll_video_emb(self, video_emb):
        video_emb = video_emb.permute(0, 1, 3, 2)
        return torch.reshape(video_emb, (video_emb.shape[0], -1, video_emb.shape[3]))

    
    def apply_pretrained_lm(self, query, query_mask):
        output = self.pretrained_lm(
            input_ids=query,
            attention_mask=query_mask
        )
        return output['last_hidden_state']
        
    
    def prepare_lang_emb(self, query, query_mask):
        # set maximum query length TODO ------ set param in config
        if query.shape[1] > 100:
            query = query[:, :100]
            query_mask = query_mask[:, :100]

        # apply pretrained language model to embed the query if specified
        if self.config['use_pretrained_lm']:
            lang_emb = self.apply_pretrained_lm(query, query_mask)
        else:
            lang_emb = self.embed(query)

        # Aloe uses an emb_dim of v_emb_dim * n_heads. Or use the emb_dim specified in the config 
        if self.config['use_pretrained_lm']:
            lang_emb = self.linear_projection_text(lang_emb)

        lang_emb = self.append_ids(lang_emb, [1, 0], 2)
        lang_emb = self.positional_encoder(lang_emb)
        return lang_emb


    def prepare_video_emb(self, video_emb):
        # shape: [batch, frames, v_emb_dim, objects]
        video_emb = self.downsample_video_emb(video_emb)

        # unroll time dimension in object dimension (only take every _ frame) - shape: [batch, objects x frames, v_emb_dim + 2]
        video_emb = self.unroll_video_emb(video_emb)
        
        # video_emb need to be projected to either the size of the language emb or the emb_size given by v_emb_dim * n_heads (As done in the Aloe paper)
        #if self.config['use_pretrained_lm'] or self.config['projection_as_in_aloe']:
        video_emb = self.linear_projection_video(video_emb)

        video_emb = self.append_ids(video_emb, [0, 1], 2)
        video_emb = self.positional_encoder(video_emb)
        return video_emb


    def forward(self, batch):
        output = self.answer_query(batch.query, batch.query_mask, batch.vft)
        return output


    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.train_config['lr'])
        sched = get_cosine_schedule_with_warmup(
            opt, 
            num_warmup_steps=self.train_config['warmup_steps'],
            num_training_steps=self.train_config['total_steps'],
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sched,
                'interval': 'step'
            }
        }