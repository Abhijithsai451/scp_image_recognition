import torch
import numpy as np
from pydantic_core.core_schema import none_schema

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import prepare_device
from zenml.steps import step
from zenml.logger import get_logger
import mlflow
import mlflow.pytorch
from typing import Tuple, Dict, Any, Optional

logger = get_logger(__name__)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@step(enable_cache=False)
def model_trainer(
        data_loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
        config_dict: Dict[str, Any],
        model_arch: str,
        loss_function: str,
        metrics_list: list,
        n_gpu:  int,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
) -> torch.nn.Module:
    """
    ZenMl Step for training the model.
    Args:
        data_loaders: Tuple of data loaders for training and validation.
        config_dict: Dictionary containing the configuration for the model.
        model_arch: Name of the model architecture to be used for training.
        loss_function: Name of the loss function to be used for training.
        metrics_list: List of metrics to be used for evaluation.
        n_gpu: Number of GPUs to be used for training.
        optimizer_config: Dictionary containing the configuration for the optimizer.
        lr_scheduler_config: Dictionary containing the configuration for the learning rate scheduler.
    Returns:
        Trained Pytorch Model
    """

    train_data_loader, valid_data_loader = data_loaders

    # Initialize the model
    model_instance = getattr(module_arch, model_arch)(**config_dict['arch']['args'])
    logger.info(f"Built the model architecture :{model_arch}")

    # Preparing the device for Training
    device, device_ids = prepare_device(n_gpu)
    # Training on MPS for Apple Silicon.
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_instance = model_instance.to(device)
    if len(device_ids) > 1:
        model_instance = torch.nn.DataParallel(model_instance, device_ids=device_ids)

    logger.info(f"Getting functions to handle loss ({loss_function}) and metrics ({metrics_list}), building optimizer ...")
    criterion = getattr(module_loss, loss_function)
    metrics = [getattr(module_metric,met) for met in metrics_list]

    trainable_params = filter(lambda p: p.requires_grad, model_instance.parameters())
    optimizer = getattr(torch.optim, optimizer_config['type'])(trainable_params, **optimizer_config['args'])
    lr_scheduler = None
    if lr_scheduler_config:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler_config['type'])(optimizer, **lr_scheduler_config['args'])

    logger.info(f"Built Optimizer and set criterion metrics, trainable_params, lr_scheduler")

    trainer = Trainer(model_instance, criterion, metrics, optimizer,
                      config = {'name': config_dict['name'], 'trainer': config_dict['trainer'],
                                     'save_dir': config_dict['trainer']['save_dir'],
                                     'epochs': config_dict['trainer']['epochs']
                                     },
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    logger.info(f"Executing trainer.train() now...")
    final_log = trainer.train()

    # MLflow Integration (Metrics and Logging)
    with mlflow.start_run(run_name:=f"Training_Run_ZenML_{mlflow.active_run().info.run_id[:8]}",nested=True):
        mlflow.log_params(config_dict)
        for metric_name, value in final_log.items():
            mlflow.log_metric(f"final_{metric_name}", value)

        mlflow.pytorch.log_model(model_instance, "model",
                                 registered_model_name=f"{config_dict['name']}_model",
                                 code_paths = ["train.py", "data_loader", "model", "trainer", "utils", "parse_config.py"])

        logger.info(f"MLflow logged to MLflow with name: {config_dict['name']}_model")

    return model_instance































