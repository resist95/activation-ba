
import decimal
import numpy as np
from sympy import Ci
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import sys
import os

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from datasets.data import Intel

class Caltech101Dataset(Dataset):

    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Resize((224,224)),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if self.transform:
            images = self.transform(self.images[idx])
            mean = torch.mean(images,[1,2])
            std = torch.std(images,[1,2])
            normalize = T.Normalize(mean,std)
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])

def main():
    i = Intel()
    train_x,train_y,test_x,test_y = i.get_data()
    train_data = IntelDataset(test_x,test_y)
    train_load = torch.utils.data.DataLoader(dataset=train_data,batch_size=100,shuffle=True)

    for a,b in train_data:
        print(torch.var(a))
    dataiter = iter(train_data)
    a,b = next(dataiter)
    print(a)

    print (a)

class IntelDataset(Dataset):

    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
        T.ToTensor()
    ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if self.transform:
            images = self.transform(self.images[idx])
            images = images.permute(1,0,2)
            normalize = T.Normalize((self.mean),(self.std))
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])
    
    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std

class Cifar10Dataset():
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        if self.transform:
            images = self.transform(self.images[idx])
            images = images.permute(1,0,2)
            normalize = T.Normalize((self.mean),(self.std))
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])
    
    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std


class MnistDataset():

    def __init__(self, images,labels):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if self.transform:
            images = self.transform(self.images[idx])
            
            mean = 0.13066047627384286
            std = 0.3081078038564622
            #normalize = T.Normalize(mean,std)
            #norm = normalize(images)
            #images = norm
        return (images,self.labels[idx])


if __name__== "__main__":

    main()
