import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ImageClassificationModel(BaseModel):

    def __init__(self, num_classes=10):
        super(ImageClassificationModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Convolutional layers with ReLU and pooling
        x = self.relu(self.conv1(x))  # Shape: [batch_size, 16, 256, 256]
        x = self.pool(x)  # Shape: [batch_size, 16, 128, 128]

        x = self.relu(self.conv2(x))  # Shape: [batch_size, 32, 128, 128]
        x = self.pool(x)  # Shape: [batch_size, 32, 64, 64]

        x = self.relu(self.conv3(x))  # Shape: [batch_size, 64, 64, 64]
        x = self.pool(x)  # Shape: [batch_size, 64, 32, 32]

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 32 * 32)  # Shape: [batch_size, 64 * 32 * 32]

        # Fully connected layers
        x = self.relu(self.fc1(x))  # Shape: [batch_size, 512]
        x = self.fc2(x)  # Shape: [batch_size, num_classes]

        return x
