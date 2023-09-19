from src.models.discriminative_model import DiscriminativeModel
from src.models.generative_model import GenerativeModel
from src.data_modules.dvd_data import DVDData
from src.data_modules.simmc2_data import Simmc2Data
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import wandb
from config.config import read_default_config, read_config, update_nested_dicts
import argparse

parser = argparse.ArgumentParser(description='Train script for OLViT')

parser.add_argument(
    '--cfg_path',
    default='config/dvd.json',
    type=str,
    help='Path to the config file of the selected checkpoint')


if __name__ == '__main__':
    wandb.finish()
    args = parser.parse_args()
    # read the default conifg and update the values with the experiment specific config
    config = read_default_config()
    experiment_config = read_config(args.cfg_path)
    config = update_nested_dicts(old_dict=config, update_dict=experiment_config)

    available_models = {
        'discriminative': DiscriminativeModel,
        'generative': GenerativeModel
    }
    data_modules = {
        'dvd': DVDData,
        'simmc2': Simmc2Data,
    }

    monitor_score = {
        'discriminative': 'val_acc',
        'generative': 'bleu4'
    }

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=monitor_score[config['model']['model_type']], mode="max",
        save_top_k=1,     
        dirpath=config["checkpoint"]["checkpoint_folder"],
        filename=config["checkpoint"]["checkpoint_file_name"],
        every_n_epochs=1  
    )

    lr_monitor_cb = LearningRateMonitor(
        logging_interval='step'
    )

    callbacks = []
    callbacks.append(checkpoint_cb)
    callbacks.append(lr_monitor_cb)
    
    wandb_logger = WandbLogger(
        offline=True,
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
        # detect_anomaly=True,
        accelerator='gpu',
        devices=[0],
        fast_dev_run=False,
        max_epochs=config['training']['epochs'],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False),
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=32,
        callbacks=callbacks
    )
    data = data_modules[config['model']['dataset']](config=config)

    if 'output_path' in config['checkpoint'].keys():
        model = available_models[config['model']['model_type']](config=config, output_path=config['checkpoint']['output_path'])
    else:
        model = available_models[config['model']['model_type']](config=config)

    trainer.fit(model, data)
