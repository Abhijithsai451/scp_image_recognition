import argparse
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.constants import MLFLOW
import os
import json
from parse_config import ConfigParser

from mlops.pipelines.ml_pipelines import train_pipeline, inference_pipeline
from mlops.steps.data_steps import load_image_data
from mlops.steps.training_steps import model_trainer
from mlops.steps.inference_steps import model_loader, inference_step, InferenceParameters, ModelDeploymentParameters

logger = get_logger(__name__)

def main(config_path:str):
    """
    This function sets up the ZenML pipelines and run them
    """
    client = Client()

    # MLFlow experimental tracker
    try:
        current_stack = client.active_stack
        if MLFLOW not in [c.flavor for c in current_stack.components.values()]:
            logger.warning("MLFlow experimental tracker is not enabled in the active stack. "
                           "Attempting to register and set up a temporary MLflow tracker "
                           "For Production, consider remote MLflow setups and configure your stack explicitly.")

            # Register a local Mlflow tracker
            os.system("zenml experiment-tracker register scp_tracker --flavor=mlflow")
            os.system(f"zenml stack register temp_mlflow_stack -e mlflow_tracker -a {current_stack.artifact_store.name}"
                      f"-o {current_stack.orchestrator.name}")

            os.system("zenml stack set temp_mlflow_stack")
            os.system("zenml integration install mlflow -y")
            client = Client()
        else:
            logger.info("MLFlow experimental tracker is found in the active stack.")

    except Exception as e:
        logger.error(f"Error setting up MLFlow experimental tracker: {e}")
        logger.warning("Please ensure ZenML is initialized (`zenml init`) and an MLflow experiment tracker is configured "
                       "(`zenml stack set --help`). Exiting.")
        return

    # 2 Load the config.json file
    config_parser = ConfigParser(config_path)
    config_dict = config_parser.config

    data_loader_params = {
        "data_dir": config_dict["data_loader"]["data_dir"],
        "aug_dir": config_dict["data_loader"]["aug_dir"],
        "batch_size": config_dict["data_loader"]["batch_size"],
        "num_workers": config_dict["data_loader"]["num_workers"],
        "validation_split": config_dict["data_loader"]["validation_split"],
        "num_aug_images": config_dict["data_loader"]["num_aug_images"],
        "shuffle": config_dict["data_loader"]['args'].get("shuffle", True)
    }

    model_trainer_params = {
        "config_dict": config_dict,
        "model_architecture": config_dict["arch"]["type"].split('.')[-1],
        "loss_function": config_dict["loss"],
        "metrics": config_dict["metrics"],
        "n_gpu": config_dict["n_gpu"],
        "optimizer_config": config_dict["optimizer"],
        "lr_scheduler_config": config_dict["lr_scheduler"]
    }


    # 3 Instantiate the pipeline and run
    logger.info("Starting the ZenML training pipeline")
    training_pipeline_instance = train_pipeline(
        data_loader_step=load_image_data(**data_loader_params),
        model_trainer_step=model_trainer(**model_trainer_params)
    )
    training_pipeline_instance.run(run_name="scp_training_run")
    logger.info("ZenML training pipeline Finished")

    # 4 Run the Inference Pipeline
    inference_params = InferenceParameters(
        model_name = f"{config_dict['name']}_model",
        model_stage= "None",
        data_path = "")

    logger.info("starting the Inference pipeline")
    inference_pipeline_instance = inference_pipeline(
        model_loader_step=model_loader(inference_params=inference_params),
        inference_step=inference_step(inference_params=inference_params)
    )
    inference_pipeline_instance.run(run_name = "scp_inference_run")
    logger.info("ZenML Inference pipeline Finished")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='SCP Image Recognition')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default="all", type=str,
                      help='indices of GPUs to enable (default: all)')

    parsed = args.parse_args()
    main(parsed.config)