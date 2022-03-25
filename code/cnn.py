
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNN_CIFAR(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_CIFAR,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16,
            kernel_size=3, stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32,
            kernel_size=3, stride=1,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=1,
            padding=1
        )
        
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.linear1 = nn.Linear(
            in_features=1024, out_features=128
        )

        self.linear2 = nn.Linear(
            in_features=128, out_features=10
        )

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1,1024)
        x = self.relu(self.linear1(x))
        x = F.softmax(self.linear2(x),dim=0)
        x = self.relu(x)
        return x

from torchsummary import summary

model = CNN_CIFAR()
summary(model,(3,32,32))


