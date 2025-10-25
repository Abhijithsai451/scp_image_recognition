import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from zenml.steps import step
from zenml.logger import get_logger
import mlflow
import mlflow.pytorch
from typing import Tuple, Dict, Any

logger = get_logger(__name__)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@step(enable_cache=False)
def model_trainer(
        config_input:ConfigParser,
        model_arch: str,
        loss_function: str,
        metrics: list,
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

    train_data_loader = config_input.init_obj('data_loader', module_data)
    valid_data_loader = train_data_loader.split_validation()

    # Initialize the model
    model = config_input.init_obj('arch', module_arch)
    logger.info(f"[INFO] model_trainer step: Built model architecture: {model.__class__.__name__}")

    # Preparing the device for Training
    device, device_ids = prepare_device(n_gpu)
    # Training on MPS for Apple Silicon.
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_instance = model.to(device)
    if len(device_ids) > 1:
        model_instance = torch.nn.DataParallel(model_instance, device_ids=device_ids)

    logger.info(f"Getting functions to handle loss ({loss_function}) and metrics ({metrics}), building optimizer ...")
    criterion = getattr(module_loss, loss_function)
    metrics = [getattr(module_metric,met) for met in metrics]

    trainable_params = filter(lambda p: p.requires_grad, model_instance.parameters())
    optimizer = config_input.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = None
    if lr_scheduler_config:
        lr_scheduler = config_input.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    logger.info(f"Built Optimizer and set criterion metrics, trainable_params, lr_scheduler")

    trainer = Trainer(model_instance, criterion, metrics, optimizer,
                      config = config_input,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    logger.info(f"Executing trainer.train() now...")
    final_log = trainer.train()

    # MLflow Integration (Metrics and Logging)
    parent_mlflow_run = mlflow.active_run()
    run_id_prefix = ""
    if parent_mlflow_run:
        run_id_prefix = parent_mlflow_run.info.run_id[:8]
    else:
        logger.warning(
            "No active parent MLflow run detected for this ZenML step. The nested run name will not include a parent run ID.")

    nested_run_name_base = f"Training_Run_ZenML_{run_id_prefix}"

    with mlflow.start_run(run_name=nested_run_name_base, nested=True) as run:
        mlflow.log_params(config_input.config)
        for metric_name, value in final_log.items():
            mlflow.log_metric(f"final_{metric_name}", value)

        mlflow.pytorch.log_model(model_instance, "model",
                                 registered_model_name=f"{config_input['name']}_model",
                                 code_paths = ["train.py", "data_loader", "model", "trainer", "utils", "parse_config.py"])

        logger.info(f"MLflow logged to MLflow with name: {config_input['name']}_model")

    return model_instance































