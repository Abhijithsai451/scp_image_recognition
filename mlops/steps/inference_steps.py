import torch
import numpy as np
import mlflow
import mlflow.pytorch
from zenml.steps import step
from zenml.logger import get_logger
from typing import Optional
from pydantic import BaseModel

logger = get_logger(__name__)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define a step for deploying/registering the model
class ModelDeploymentParameters(BaseModel):
    model_name:str
    model_version: Optional[str] = None

@step
def model_deployer(
        model: torch.nn.Module,
        params: ModelDeploymentParameters
) -> None:
    """
    Register the Trained Model in the Mlflow Model Registry
    """
    logger.info(f"Registering the model '{params.model_name}' in the Mlflow Model Registry")
    with mlflow.start_run(run_name:=f"Model_RegistrationL_{params.model_name}",nested=True):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=params.model_name)
    logger.info(f"Model '{params.model_name}' registered successfully in the Model Registry")

# Define a step for loading a model from MLFlow
class InferenceParameters(BaseModel):
    model_name:str
    model_stage: str = "Production"
    data_path: str

@step
def model_loader(params: InferenceParameters)-> torch.nn.Module:
    """
    Loads the model form the MLflow model Registry
    """
    model_uri = f"models:/{params.model_name}/{params.model_stage}"
    logger.info(f"Loading the model from the MLflow model registry: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    logger.info(f"Model loaded successfully from the MLflow model registry")
    return model

@step
def inference_step(
        model: torch.nn.Module,
        params: InferenceParameters
) -> torch.Tensor:
    """
    Performs inference using the loaded model on the new data.
    """
    logger.info(f"Performing inference on the data from: {params.data_path}")

    test_input = torch.randn(1, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        output = model(test_input)

    output_np = output.cpu().numpy()
    logger.info(f"Inference completed successfully : output.shape is {output_np.shape}")

    return output























