import torch
from zenml.steps import step
from typing import Tuple
from data_loader.data_loaders import ImageDataLoader

@step
def load_image_data(
        data_dir: str,
        aug_dir: str,
        batch_size: int,
        num_workers: int,
        validation_split: float,
        num_aug_images: int,
        shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    ZenML Step for loading and preprocessing the data.
    Args:
        data_dir: Path to the directory containing the data.
        aug_dir: Path to the directory where the augmented data will be stored.
        batch_size: Batch size for the data loaders.
        num_workers: Number of workers for the data loaders.
        validation_split: Fraction of the data to be used for validation.
        num_aug_images: Number of augmented images to be generated for each image.
        shuffle: Whether to shuffle the data.

    Returns:
        A tuple containing the training and validation data loaders.
    """
    data_loader_instance = ImageDataLoader(data_dir, aug_dir, batch_size, num_workers, shuffle, validation_split, num_aug_images)

    train_data_loader = data_loader_instance
    valid_data_loader = data_loader_instance.split_validation()

    return train_data_loader, valid_data_loader


