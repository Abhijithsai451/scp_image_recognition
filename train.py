import argparse
import collections

import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    logger.info("[INFO] In MAIN method: Initializing data_loader variable...")
    data_loader = config.init_obj('data_loader', module_data)
    logger.info("[INFO] Executing data_loader")
    valid_data_loader = data_loader.split_validation()
    logger.info("[INFO] Executing valid_data_loader")
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info("[INFO] Built the model architecture : ")

    logger.info("[INFO] Preparing for GPU training, if available, else CPU...")
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # For training with GPU (NVIDIA etc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For Training with GPU in Apple Silicon.
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info("[INFO] Getting functions to handle loss and metrics, building optimizer ...")
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    logger.info("[INFO] Built optimizer and set criterion  metrics, trainable_params, lr_scheduler")

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    logger.info("[INFO] Executing trainer.train() now...")
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='SCP Image Recognition')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='all', type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
