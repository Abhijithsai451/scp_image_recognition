import argparse
import collections

from zenml.client import Client
from zenml.enums import StackComponentType
from zenml.logger import get_logger
from zenml.integrations.constants import MLFLOW
import os
import json
from parse_config import ConfigParser

from mlops.pipelines.ml_pipelines import train_pipeline, inference_pipeline
from mlops.steps.data_steps import load_image_data
from mlops.steps.training_steps import model_trainer
from mlops.steps.inference_steps import model_loader, inference_step, InferenceParameters, ModelDeploymentParameters
from mlops.materializer.config_materializer import ConfigParserMaterializer


logger = get_logger(__name__)

def main(config: ConfigParser):
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
            tracker_name = "scp_tracker"
            temp_stack_name = "scp_mlflow_stack"
            try:
                experimental_tracker = client.get_stack_component(
                    name_id_or_prefix = tracker_name,
                    component_type = StackComponentType.EXPERIMENT_TRACKER,
                )
                logger.info(f"MLFlow experimental tracker '{tracker_name}' already exists.")
            except KeyError:
                experimental_tracker = client.create_stack_component(
                    name = tracker_name,
                    component_type = StackComponentType.EXPERIMENT_TRACKER,
                    flavor = MLFLOW,
                    configuration={}
                )
                logger.info(f"Experimental Tracker '{tracker_name}' created successfully.")

            artifact_store = current_stack.artifact_store
            orchestrator = current_stack.orchestrator

            # Defining the components of the temporary stack
            temp_stack_components = {
                "experiment_tracker": experimental_tracker.id,
                "artifact_store": artifact_store.id,
                "orchestrator": orchestrator.id
            }
            try:
                temp_stack = client.get_stack(name_id_or_prefix=temp_stack_name)

                curr_temp_stack_components = {
                    comp.type.value: str(comp.id) for comp in temp_stack.components.values()
                }

                if curr_temp_stack_components != {comp_type.value: comp_id for comp_type, comp_id
                                                  in temp_stack_components.items()}:
                    client.update_stack(
                        name= temp_stack_name,
                        components=temp_stack_components
                    )
                    logger.info(f"Updated the temporary stack '{temp_stack_name}' with the new components.")
                else:
                    logger.info(f"Temporary stack '{temp_stack_name}' already exists and is configured correctly.")
            except KeyError:
                temp_stack = client.create_stack(
                    name= temp_stack_name,
                    components=temp_stack_components
                )
                logger.info(f"Stack '{temp_stack_name}' created successfully.")

            # Set the created/retrieved stack as active
            client.activate_stack(temp_stack.name)
            logger.info(f"Active stack set to '{temp_stack.name}'.")

            client = Client()
    except Exception as e:
        logger.error(f"Error occurred while setting up the MLFlow experimental tracker: {e}")
        logger.warning("Please ensure ZenML is initialized (`zenml init`) and an MLflow experiment tracker is configured "
                       "(`zenml stack set --help`). Exiting.")

        return

    # 2 Load the config.json file
    config_dict = config.config
    pipeline_resume_path = config._resume_path
    model_trainer_params = {
        #"config_dict": config,
        "model_arch": config["arch"]["type"].split('.')[-1],
        "loss_function": config["loss"],
        "metrics": config["metrics"],
        "n_gpu": config["n_gpu"],
        "optimizer_config": config["optimizer"],
        "lr_scheduler_config": config["lr_scheduler"]
    }
    # 3 Instantiate the pipeline and run
    logger.info("Starting the ZenML training pipeline")
    training_pipeline_instance = train_pipeline(
        config_dict=config_dict,
        resume_path=pipeline_resume_path,
        model_trainer_params=model_trainer_params
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
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="all", type=str,
                      help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)