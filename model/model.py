import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

"""
class ImageClassificationModel(BaseModel):

    def __init__(self, num_classes=10):
        super(ImageClassificationModel, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        #Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

        #dropout layers
        self.dropout = nn.Dropout2d(p=0.2)  # Spatial dropout for conv layers
        self.dropout_fc = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Convolutional layers with ReLU and pooling
        # Convolution Block 1
        x = self.relu(self.conv1(x))  # Shape: [batch_size, 16, 256, 256]
        x = self.bn1(x)
        x = self.pool(x)  # Shape: [batch_size, 16, 128, 128]
        x = self.dropout(x)

        # Convolution Block 2
        x = self.relu(self.conv2(x))  # Shape: [batch_size, 32, 128, 128]
        x = self.bn2(x)
        x = self.pool(x)  # Shape: [batch_size, 32, 64, 64]
        x = self.dropout(x)

        # Convolution Block 3
        x = self.relu(self.conv3(x))  # Shape: [batch_size, 64, 64, 64]
        x = self.bn3(x)
        x = self.pool(x)  # Shape: [batch_size, 64, 32, 32]

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 32 * 32)  # Shape: [batch_size, 64 * 32 * 32]

        # Fully connected layers
        x = self.relu(self.fc1(x))  # Shape: [batch_size, 512]
        x = self.dropout_fc(x)
        x = self.fc2(x)  # Shape: [batch_size, num_classes]

        return x

"""

class ImageClassificationModel(BaseModel):
    def __init__(self, num_classes=10):
        super(ImageClassificationModel, self).__init__()

        # Convolutional layers (matches table exactly)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Conv-1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Conv-4
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # Conv-7
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Conv-9

        # Max pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool-3
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool-6
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer
        self.fc = nn.Linear(64 * 4 * 4, num_classes)  # Linear-11

        # Dropout (not in table but specified in requirements)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        # Conv-1 -> ReLU-2 -> MaxPool-3
        x = F.relu(self.conv1(x))  # [batch, 16, 32, 32]
        x = self.pool1(x)  # [batch, 16, 16, 16]

        # Conv-4 -> ReLU-5 -> MaxPool-6
        x = F.relu(self.conv2(x))  # [batch, 32, 16, 16]
        x = self.pool2(x)  # [batch, 32, 8, 8]

        # Conv-7 -> ReLU-8
        x = F.relu(self.conv3(x))  # [batch, 32, 8, 8]

        # Conv-9 -> ReLU-10
        x = F.relu(self.conv4(x))  # [batch, 64, 8, 8]
        x = self.pool3(x)

        # Flatten and fully connected
        x = x.view(-1, 64 * 4 * 4)  # Flatten to [batch, 1024]
        x = self.dropout(x)  # Optional dropout
        x = self.fc(x)  # [batch, 10]

        return x
"""

class ImageClassificationModel(BaseModel):
    def __init__(self, num_classes=10):
        super(ImageClassificationModel, self).__init__()

        # Convolutional Blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3,stride =1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3,stride =1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier layers (fully connected)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x
    """