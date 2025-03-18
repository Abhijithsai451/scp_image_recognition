import os

import torch
from torchvision import datasets
import torch.multiprocessing as mp
from base import BaseDataLoader
from data_transformers import data_transform
from data_transformers.data_augment import augment_data

num_workers = min(4, os.cpu_count())


class ImageDataLoader(BaseDataLoader):
    print(' [DEBUG] data_loaders.py BaseDataLoader', BaseDataLoader)
    """
       Loading the data from the given path
       """

    def __init__(self, data_dir, aug_dir, batch_size, shuffle=True, validation_split=0.0, num_aug_images=1, training=True):
        trsfm = data_transform.image_transforms.get('train')
        self.data_dir = data_dir
        self.aug_dir = aug_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_aug_images = num_aug_images


        if not os.path.isdir(aug_dir):
            dataset = datasets.ImageFolder(root=data_dir, transform=trsfm)
            dataset_splits = torch.utils.data.random_split(dataset, [len(dataset) // num_workers] * num_workers)

            # Run multiprocessing for augmentation
            with mp.Pool(num_workers) as pool:
                pool.starmap(augment_data, [(i, dataset_splits[i], aug_dir,num_aug_images, True) for i in range(num_workers)])

            # data_augment.augment_data(dataset_splits, aug_dir, True)
            self.dataset = datasets.ImageFolder(root=aug_dir, transform=trsfm)

        else:
            self.dataset = datasets.ImageFolder(root=aug_dir, transform=trsfm)
            print("Augmentation needs not to be done")

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
