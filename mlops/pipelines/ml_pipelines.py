from pathlib import Path
from typing import Dict, Any, Optional

from zenml.pipelines import pipeline
from zenml.logger import get_logger

from mlops.steps.data_steps import load_image_data
from mlops.steps.inference_steps import InferenceParameters, model_loader, inference_step
from mlops.steps.training_steps import model_trainer
from parse_config import ConfigParser

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def train_pipeline(

        config_dict: Dict[str, Any],
        resume_path: Optional[Path],
        model_trainer_params: Dict[str, Any]

):
    """
    A pipeline to load the image data, train the model and log it to the MLFlow
    """

    config_object_for_steps = load_image_data(config_dict=config_dict, resume_path=resume_path
)
    trained_model = model_trainer(config_input=config_object_for_steps, **model_trainer_params
)

    return trained_model

@pipeline
def inference_pipeline(
        inference_params: InferenceParameters,
):
    """
    Pipeline to load a trained model from MLflow registry and perform inference
    """
    loaded_model = model_loader(params=inference_params)
    prediction = inference_step(model=loaded_model,params=inference_params)

    return prediction