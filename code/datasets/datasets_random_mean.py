import decimal
import numpy as np
from sympy import Ci
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))


from datasets.datasets import Cifar10Dataset, IntelDataset, MnistDataset
from datasets.data import CIFAR10, Intel, MNIST


class IntelDataset_random_mean(Dataset):

    def __init__(self,images,labels,train=False):
        self.mean = torch.zeros(3)
        self.std = torch.zeros(3)
        self.train = train
        self.images = np.array(images)
        self.labels = labels
        self.transform = T.Compose([
        T.ToTensor()
    ])

    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std
    
    def get_mean_std(self):
        return self.mean,self.std

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        images = self.transform(self.images[idx])
        images = torch.transpose(images,0,1)
        if self.train:
            mean = torch.mean(images,[1,2])
            std = torch.std(images,[1,2])
            normalize = T.Normalize(mean,std)
            norm = normalize(images)
            images = norm
            self.mean += mean
            self.std += std
        else:
            normalize = T.Normalize(self.mean,self.std)    
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])

i = Intel()
train_x,train_y,test_x,test_y = i.get_data()
train_data = IntelDataset_random_mean(test_x,test_y,train=True)
train_load = torch.utils.data.DataLoader(dataset=train_data,batch_size=100,shuffle=True)

class Cifar10Dataset():
    def __init__(self,images,labels,train=False):
        self.images = images
        self.labels = labels
        self.train = train
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
    
    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std
    
    def get_mean_std(self):
        return self.mean,self.std

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        images = self.transform(self.images[idx])
        if self.train:
            mean = torch.mean(images,[1,2])
            std = torch.std(images,[1,2])
            normalize = T.Normalize(mean,std)
            norm = normalize(images)
            images = norm
            self.mean += mean
            self.std += std
        else:
            normalize = T.Normalize(self.std,self.mean)
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])


class MnistDataset():

    def __init__(self, images,labels,train = False):
        self.images = images
        self.labels = labels
        self.train = train
        self.transform = T.Compose([
            T.ToTensor()
        ])
    
    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std
    
    def get_mean_std(self):
        return self.mean,self.std

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        images = self.transform(self.images[idx])
        if self.train :
            mean = images.mean()
            std = images.std()
            normalize = T.Normalize(mean,std)
            norm = normalize(images)
            images = norm
            self.mean += mean
            self.std += std
        else:
            normalize = T.Normalize(mean,std)
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])
