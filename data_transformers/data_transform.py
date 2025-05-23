from torchvision.transforms import v2
import torch

"""
Transforms the images to a specified shape (256x256) so as to give input to the ALEXNET
"""
image_transforms = {
    "train": v2.Compose([
        #v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": v2.Compose([
        v2.Resize((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

"""
Augments the input images to increase the data and to make the model more robust
"""

image_augment = {
    "train": v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.ToDtype(torch.float32, scale=True)
    ]),
    "test": v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.ToDtype(torch.float32, scale=True)
    ])

}
