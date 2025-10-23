from zenml.pipelines import pipeline
from zenml.logger import get_logger

from mlops.steps.data_steps import load_image_data
from mlops.steps.training_steps import model_trainer
from mlops.steps.inference_steps import model_loader, inference_step

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def train_pipeline(
        data_loader_step,
        model_trainer_step,
):
    """
    A pipeline to load the image data, train the model and log it to the MLflow
    """
    train_dl, valid_dl = data_loader_step()
    trained_model = model_trainer_step(data_loaders=(train_dl, valid_dl))

    return trained_model

@pipeline
def inference_pipeline(
        model_loader_step,
        inference_step,
):
    """
    Pipeline to load a trained model from MLflow registry and perform inference
    """
    loaded_model = model_loader_step()
    prediction = inference_step(model=loaded_model)

    return prediction