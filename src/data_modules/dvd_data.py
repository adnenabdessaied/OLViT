import pytorch_lightning as pl
import src.utils.dvd_codebase.data.data_handler as dh
from  src.utils.dvd_codebase.configs.configs import *
from transformers import AutoTokenizer
import os

class DVDData(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        args.batch_size = config['training']['batch_size']
        args.fea_dir = config['datamodule']['fea_dir']
        args.data_dir = config['datamodule']['data_dir']
        pretrained_lm_name = config['model']['pretrained_lm_name']

        # load dialogues 
        self.train_dials, self.train_vids = dh.load_dials(args, "train")
        self.val_dials, self.val_vids = dh.load_dials(args, "val")
        self.test_dials, self.test_vids = dh.load_dials(args, "test")
        
        # get vocabulary 
        self.vocab, self.answer_list = dh.get_vocabulary(self.train_dials, args)
        # self.answer_list =     ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'False', 'True', 'blue', 'brown', 'cone', 'cube', 'cyan', 'cylinder', 'flying', 'flying,rotating', 'flying,rotating,sliding', 'flying,sliding', 'gold', 'gray', 'green', 'large', 'medium', 'metal', 'no action', 'purple', 'red', 'rotating', 'rotating,sliding', 'rubber', 'sliding', 'small', 'sphere', 'spl', 'yellow']

        train_vft = dh.load_video_features(args, self.train_vids)
        val_vft = dh.load_video_features(args, self.val_vids)
        test_vft = dh.load_video_features(args, self.test_vids)

        # create tokenizer
        if pretrained_lm_name != '':
            tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_name)
            pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) 
            self.vocab['<blank>'] = pad_token_id
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            tokenizer = None

        # load data
        self.train_dials = dh.create_dials(self.train_dials, self.vocab, self.answer_list, train_vft, args, tokenizer=tokenizer)
        self.val_dials = dh.create_dials(self.val_dials, self.vocab, self.answer_list, val_vft, args, tokenizer=tokenizer)
        self.test_dials = dh.create_dials(self.test_dials, self.vocab, self.answer_list, test_vft, args, tokenizer=tokenizer)
        
    
    def train_dataloader(self):
        dl, _ = dh.create_dataset(self.train_dials, self.vocab, "train", args)
        return dl

    def val_dataloader(self):
        dl, _ = dh.create_dataset(self.val_dials, self.vocab, "val", args)
        return dl
    
    def test_dataloader(self):
        dl, _ = dh.create_dataset(self.test_dials, self.vocab, "test", args)
        return dl


