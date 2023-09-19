from src.models.discriminative_model import DiscriminativeModel
from src.models.generative_model import GenerativeModel
from src.data_modules.dvd_data import DVDData
from src.data_modules.simmc2_data import Simmc2Data
from src.data_modules.avsd_data import AvsdData
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import wandb
from config.config import read_default_config, read_config, update_nested_dicts
import argparse


parser = argparse.ArgumentParser(description='Test script for OLViT')
parser.add_argument(
    '--ckpt_path',
    type=str,
    help='Path to the checkpoint to be tested')

parser.add_argument(
    '--cfg_path',
    type=str,
    help='Path to the config file of the selected checkpoint')


if __name__ == '__main__':
    wandb.finish()
    args = parser.parse_args()

    chkpt_path = args.ckpt_path

    # read the default conifg and update the values with the experiment specific config
    config = read_default_config()
    experiment_config = read_config(args.cfg_path)
    config = update_nested_dicts(old_dict=config, update_dict=experiment_config)

    if 'output_path' not in config['checkpoint'].keys():
        raise Exception('no output path provided in config (full path for disc model only path to output folder for gen. model)')

    available_models = {
        'discriminative': DiscriminativeModel,
        'generative': GenerativeModel
    }
    data_modules = {
        'dvd': DVDData,
        'simmc2': Simmc2Data,
    }
    
    wandb_logger = WandbLogger(
        entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        group=config['wandb']['group'],
        tags=config['wandb']['tags'],
        project=config['wandb']['project'],
        config=config
    )

    if config['training']['seed'] != None:
        pl.seed_everything(config['training']['seed'])

    trainer = Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=[0]
    )
    data = data_modules[config['model']['dataset']](config=config)

    model = available_models[config['model']['model_type']](config=config, output_path=config['checkpoint']['output_path'])
    trainer.test(model=model, ckpt_path=chkpt_path, dataloaders=data)
