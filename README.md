# SCP Image Recognition Project

This repository hosts an image recognition project leveraging MLOps principles with ZenML and MLflow for robust and reproducible machine learning workflows. The project is designed to handle image classification tasks, providing pipelines for training, experiment tracking, and inference.

## 1. About the Project

This project implements an image recognition system with a focus on MLOps best practices. It utilizes ZenML to orchestrate end-to-end machine learning pipelines, encompassing data loading, model training, and inference. Key features include:

*   **Modular Pipeline Design**: Clearly separated steps for data preparation, model training, and inference using ZenML pipelines.
*   **Experiment Tracking**: Integration with MLflow to log experiment runs, metrics, parameters, and artifacts, ensuring full traceability and reproducibility of model development.
*   **Configurable Training**: Training parameters are managed via a `config.json` file, allowing easy modification and experimentation.
*   **Checkpointing and Resumption**: The training process supports saving and resuming from checkpoints.
*   **Scalability**: Designed to potentially leverage multiple GPUs for training.

## 2. Technologies Used

The project is built upon a modern Python-based machine learning stack:

*   **Python**: The primary programming language.
*   **ZenML**: An extensible, open-source MLOps framework used for defining and orchestrating machine learning pipelines.
*   **MLflow**: An open-source platform for managing the ML lifecycle, primarily used here for experiment tracking.
*   **PyTorch**: A powerful open-source machine learning framework for deep learning tasks, used for building and training the neural network models.
*   **Numpy**: Fundamental package for numerical computation in Python.
*   **Pandas**: Data manipulation and analysis library.
*   **Pillow**: Image processing library.
*   **Alembic**: Database migration tool (often used with SQLAlchemy).
*   **SQLAlchemy**: SQL toolkit and Object-Relational Mapper.
*   **Matplotlib**: Plotting library (potentially for visualizing results or training metrics).
*   **TensorBoard**: A visualization toolkit for machine learning experimentation (enabled via configuration).

## 3. Execution

This section guides you through setting up the environment and running the project's pipelines.

### Environment Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and Activate a Conda Environment**:
    It is highly recommended to use `condavenv` for managing dependencies.
    ```bash
    conda create -n scp_env python=3.9
    conda activate scp_env
    ```

3.  **Install Dependencies**:
    While a `requirements.txt` is not provided here, you will need to install the core libraries.
    ```bash
    pip install zenml mlflow torch torchvision numpy pandas pillow matplotlib alembic sqlalchemy
    ```
    *Note: Depending on your system and PyTorch version, you might need specific installation instructions for `torch` (e.g., for CUDA or MPS support).*

4.  **Initialize ZenML**:
    ZenML requires initialization in your project directory.
    ```bash
    zenml init
    ```

5.  **Set up MLflow (Optional, but Recommended)**:
    The `main.py` script attempts to set up a temporary MLflow tracker. For persistent tracking or production environments, you should configure ZenML with an MLflow experiment tracker explicitly:
    ```bash
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml stack register mlflow_stack -o default -a default -e mlflow_tracker
    zenml stack set mlflow_stack
    ```
    This ensures your MLflow runs are stored and accessible.

### Running the Pipelines

The `main.py` script serves as the entry point for executing the training and inference pipelines.

1.  **Prepare your Data**:
    Ensure your dataset is placed in the location specified in `config.json` (e.g., `data/input_dataset` and `data/aug_dataset`). The `config.json` file dictates how your data is loaded and augmented.

2.  **Configure Training Parameters**:
    Review and modify the `config.json` file located in the project root. This file defines:
    *   Model architecture (`arch`)
    *   Data loading parameters (`data_loader`)
    *   Optimizer settings (`optimizer`)
    *   Loss function (`loss`)
    *   Metrics to track (`metrics`)
    *   Learning rate scheduler (`lr_scheduler`)
    *   Trainer specifics like epochs, save directory, and TensorBoard logging (`trainer`)

3.  **Execute the Main Script**:
    Run the `main.py` script from your terminal. By default, it uses `config.json`.
    ```bash
    python main.py -c config.json
    ```
    You can specify a different configuration file or resume training from a checkpoint:
    ```bash
    python main.py --config path/to/your/config.json --resume path/to/your/checkpoint.pth
    ```
    The script will:
    *   Set up a ZenML stack with an MLflow experiment tracker.
    *   Execute the `train_pipeline` to train your model.
    *   Execute the `inference_pipeline` to perform inference using the trained model.

## 4. Results

Upon successful execution of the `main.py` script, the following results and artifacts will be generated:

*   **MLflow UI**: You can view all experiment runs, logged parameters, metrics (e.g., accuracy, loss, F1-score), and saved model artifacts by launching the MLflow UI.
    ```bash
    mlflow ui
    ```
    Navigate to `http://localhost:5000` (or the address indicated by MLflow) in your web browser. You will find runs for both the training and inference pipelines.

*   **Saved Checkpoints**: Model checkpoints, including the best performing model (`model_best.pth`), will be saved in the directory specified by `trainer.save_dir` in your `config.json` (default: `saved/`).

*   **TensorBoard Logs**: If `tensorboard` is set to `true` in your `config.json`, training metrics and visualizations will be logged to TensorBoard. You can launch TensorBoard to inspect these logs:
    ```bash
    tensorboard --logdir=runs/
    ```
    (Adjust `runs/` to your actual log directory, usually specified in `config.log_dir` and managed by the `TensorboardWriter`).

The MLflow UI will provide a comprehensive overview of each run, allowing you to compare different experiments, track model performance over epochs, and access saved models for deployment.
