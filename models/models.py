import torch
import torch.nn as nn 
import torch.nn.functional as F

from typing import Tuple, List, Union

from .utils import register_model

__all__ = ['BasicCNN', 'FullyConnected']

@register_model()
class BasicCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN).

    Args:
        n_channels (int): The number of channels of input image.
        n_classes (int): The number of output classes.
    """
    def __init__(self, input_shape: List[int], n_classes: int) -> None:
        super(BasicCNN, self).__init__()
        n_channels = input_shape[0]
        self.conv1 = nn.Conv2d(n_channels, 6, 5)  
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc1 = nn.Linear(16, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.adaptive_pool(x) 
        x = torch.flatten(x, 1)    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@register_model()
class FullyConnected(nn.Module):
    """
    A simple Fully Connected (FC) neural network.

    Args:
        input_shape (Union[Tuple[int, int, int], List[int]]): The shape of the input tensor as (n_channels, height, width).
        n_classes (int): The number of output classes.
    """
    def __init__(self, input_shape: Union[Tuple[int, int, int], List[int]], n_classes: int):
        super(FullyConnected, self).__init__()
        n_channels, height, width = input_shape
        self.fc1 = nn.Linear(n_channels * height * width, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x