from meldataset import build_dataloader
from optimizers import build_optimizer
from utils import *
from models import build_model
from trainer import Trainer

import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import click

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('-p', '--config_path', default='./Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.mkdir(log_dir)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)

    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config=config.get('dataset_params', {}),
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      dataset_config=config.get('dataset_params', {}))

    model = build_model(model_params=config['model_params'] or {})

    scheduler_params = {
            "max_lr": float(config['optimizer_params'].get('lr', 5e-4)),
            "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
            "epochs": epochs,
            "steps_per_epoch": len(train_dataloader),
        }

    model.to(device)
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})

    blank_index = train_dataloader.dataset.text_cleaner.word_index_dictionary[" "] # get blank index

    criterion = build_criterion(critic_params={
                'ctc': {'blank': blank_index},
        })

    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    logger=logger)

    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    for epoch in range(1, epochs+1):
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.4f' % (key, value))
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_attn', plot_image(v), epoch)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))
            
    return 0

if __name__=="__main__":
    main()