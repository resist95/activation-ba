#default dependencies
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
import sys
import os

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from datasets.data import CIFAR10,Intel,MNIST


def main():
    i = CIFAR10(test_size=0.1,run='validate')
    train_x,train_y = i.get_data(what_is_needed='train')
    train_data = CustomDataset(train_x,train_y)
    train_data.set_mean_std((0.5,0.5,0.5),(0.5,0.5,0.5))
    train_load = torch.utils.data.DataLoader(dataset=train_data,batch_size=100,shuffle=True)

    dataiter = iter(train_data)
    a,b = next(dataiter)

class CustomDataset(Dataset):
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

if __name__== "__main__":

    main()
