import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
#Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
class datasets:
    
    def __init__(self):
        self.cifar_data = torchvision.datasets.CIFAR10('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/CIFAR',transform=ToTensor())
        self.caltech_data = torchvision.datasets.Caltech101('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/CALTECH')
        self.mnsit_data = torchvision.datasets.MNIST('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/MNIST')
        self.stl_data = torchvision.datasets.STL10('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/STL')

    def load
