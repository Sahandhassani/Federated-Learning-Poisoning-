import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR_CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # -------- Block 1 --------
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1   = nn.BatchNorm2d(32)

        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2   = nn.BatchNorm2d(32)

        # -------- Block 2 --------
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1   = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2   = nn.BatchNorm2d(64)

        # -------- Block 3 --------
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(128)

        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # -------- Fully Connected --------
        # After 3 poolings: 32x32 → 16x16 → 8x8 → 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.fc3(x)

        return logits   # CrossEntropyLoss applies Softmax internally
