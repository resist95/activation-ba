import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

cifar_data_train = torchvision.datasets.CIFAR10('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/cifar-10',
train=True, transform=ToTensor(),
)

cifar_data_test = torchvision.datasets.CIFAR10('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/cifar-10',
train=False, transform=ToTensor(),
)

train_dataloader = DataLoader(cifar_data_train, batch_size=64)
test_dataloader = DataLoader(cifar_data_test,batch_size=64)



