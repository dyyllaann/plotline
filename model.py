import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms


# %%
class NN(nn.Module):
    def __init__(self, arr=[]):
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 30 * 3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# %%
class SimpleCNN(nn.Module):
    def __init__(self, arr=[]):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1568, 2)

    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


# %%
basic_transformer = transforms.Compose([transforms.ToTensor()])

norm_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""Add random data augmentation to the transformer"""
aug_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomAffine(degrees = 5, shear = 10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# %%
class DeepCNN(nn.Module):
    def __init__(self, arr=[]):
        super(DeepCNN, self).__init__()

        num_channels = 3
        size = 224
        layers = []

        for a in arr:
          if a == "pool":
            layers.append(nn.MaxPool2d(kernel_size = 2))
            size //= 2
          else:
            layers.append(nn.Conv2d(num_channels, a, kernel_size = 3))
            layers.append(nn.ReLU())
            num_channels = a
            size -= 2
        
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels * size * size, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
