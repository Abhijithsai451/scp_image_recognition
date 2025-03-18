import os

import torch

from torchvision.utils import save_image

from data_transformers.data_transform import image_augment

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

"""
def augment_data(dataset, aug_dir, aug_exist):
    if (aug_exist):
        aug_transform = image_augment.get('train')
        for img_idx, (image, label) in enumerate(dataset):
            class_name = dataset.classes[label]
            class_path = os.path.join(aug_dir, class_name)
            os.makedirs(class_path, exist_ok=True)

            image = image.to(device)

            for i in range(5):  # Create 5 augmented versions per image
                aug_image = aug_transform(image)
                aug_path = os.path.join(class_path, f"aug_{img_idx}_{i}.jpg")
                save_image(aug_image, aug_path)

        with open(aug_exist, "w") as f:
            f.write("Augmented data exists.")
    else:
        print("Augmented dataset already exists. Will proceed to the training directly")
"""



def augment_data(worker_id, dataset_split, aug_dir,num_aug_images,aug_exist):

    if (aug_exist):
        aug_transform = image_augment.get('train')
        for img_idx, (image, label) in enumerate(dataset_split):
            class_name = dataset_split.dataset.classes[label]
            class_path = os.path.join(aug_dir, class_name)
            os.makedirs(class_path, exist_ok=True)

            image = image.to(device)

            for i in range(num_aug_images):  # Create number of augmented versions per image
                aug_image = aug_transform(image)
                aug_path = os.path.join(class_path, f"aug_{img_idx}_{i}.jpg")
                save_image(aug_image, aug_path)
        print(f"Worker {worker_id} finished processing.")
        with open(aug_exist, "w") as f:
            f.write("Succesfully augmented data from the Input Image Dataset.")
    else:
        print("Augmented dataset already exists. Will proceed to the training directly")
