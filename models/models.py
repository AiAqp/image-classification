import torch
import torch.nn as nn 
import torch.nn.functional as F

from .utils import register_model

__all__ = ['BasicCNN', 'FullyConnected']

@register_model()
class BasicCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) model.
    """
    def __init__(self, n_classes: int) -> None:
        super(BasicCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@register_model()
class FullyConnected(nn.Module):
    """
    A simple Fully Connected (FC) model.
    """
    def __init__(self, n_classes: int):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x