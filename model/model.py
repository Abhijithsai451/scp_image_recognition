import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ImageClassificationModel(BaseModel):
    print(' [DEBUG] model.py ImageClassificationModel file ',BaseModel )
    '''def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    '''


    def __init__(self, num_classes=10):
        super(ImageClassificationModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Adjust based on the output size of the last conv layer
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: [batch_size, 3, 256, 256]

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
