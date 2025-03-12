from torchvision import transforms
"""
Creating the transform to standardize the input images
"""
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
    "test": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
}