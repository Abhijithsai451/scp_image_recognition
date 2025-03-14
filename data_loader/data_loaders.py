import os
from torchvision import datasets

from base import BaseDataLoader
from data_transformers import data_transform
from data_transformers import data_augment



class ImageDataLoader(BaseDataLoader):
    """
       Loading the data from the given path
       """

    def __init__(self, data_dir, aug_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = data_transform.image_transforms.get('train')
        self.data_dir = data_dir
        self.aug_dir = aug_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not os.path.isdir(aug_dir):
            dataset = datasets.ImageFolder(root=data_dir, transform=trsfm)
            data_augment.augment_data(dataset, aug_dir, True)
            self.dataset = datasets.ImageFolder(root=aug_dir, transform=None)
        else :
            self.dataset = datasets.ImageFolder(root=aug_dir, transform=None)
            print("Augmentation needs not to be done")

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
