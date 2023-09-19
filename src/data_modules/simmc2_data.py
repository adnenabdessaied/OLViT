import pytorch_lightning as pl
from src.utils.simmc2_dataset.dataloader_dvd_model import Simmc2Dataset, VisualFeatureLoader
from transformers import AutoTokenizer
import argparse
import os
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='', help="Path to train file")
    parser.add_argument("--dev_file", default='', help="Path to dev file")
    parser.add_argument("--devtest_file", default='', help="Path to devtest file")
    parser.add_argument(
        "--visual_feature_path", default=None, help="Path to visual features"
    )
    parser.add_argument(
        "--visual_feature_size",
        type=int,
        default=516,
        help="Size of the visual features",
    )
    parser.add_argument(
        "--max_turns", type=int, default=5, help="Number of turns in history"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum length in utterance"
    )
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=True)
    args = parser.parse_args()
    return args



class Simmc2Data(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.args = parse_arguments()
        self.args.train_file = os.path.join(config['datamodule']['data_dir'], 'simmc2.1_ambiguous_candidates_dstc11_train.json')
        self.args.dev_file = os.path.join(config['datamodule']['data_dir'], 'simmc2.1_ambiguous_candidates_dstc11_dev.json')
        self.args.devtest_file = os.path.join(config['datamodule']['data_dir'], 'simmc2.1_ambiguous_candidates_dstc11_devtest.json')
        self.args.teststd_file = os.path.join(config['datamodule']['data_dir'], 'simmc2.1_dials_dstc11_dev.json')
        self.args.visual_feature_path = config['datamodule']['fea_dir']
        pretrained_lm_name = config['model']['pretrained_lm_name']
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_name)
        self.feature_loader = VisualFeatureLoader(
            feature_path=self.args.visual_feature_path,
            feature_size=self.args.visual_feature_size
        )
        self.config = config

    def train_dataloader(self):
        dataset = Simmc2Dataset(
            tokenizer=self.tokenizer,
            feature_loader=self.feature_loader,
            load_path=self.args.train_file,
            args=self.args
        )
        dl = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )
        return dl

    def val_dataloader(self):
        dataset = Simmc2Dataset(
            tokenizer=self.tokenizer,
            feature_loader=self.feature_loader,
            load_path=self.args.dev_file,
            args=self.args,
        )
        dl = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        return dl

    def test_dataloader(self):
        dataset = Simmc2Dataset(
            tokenizer=self.tokenizer,
            feature_loader=self.feature_loader,
            load_path=self.args.devtest_file,
            args=self.args,
        )
        dl = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        return dl
