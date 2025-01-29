from torchvision import datasets, transforms
from base import BaseDataLoader
from data_transform import data_transform

class ImageDataLoader(BaseDataLoader):
    """
    Loading the data from the given path
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = data_transform.image_transforms
        self.data_dir = data_dir


        #self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)

        self.dataset = datasets.ImageFolder(root=data_dir,transform = trsfm)
        self.batch_size = batch_size
        self.shuffle = shuffle


        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


