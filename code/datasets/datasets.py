#default dependencies
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
import sys
import os

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

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
            images = images.permute(1,0,2) #umformatierung von h,c,w in c,h,w
            normalize = T.Normalize((self.mean),(self.std)) #normalisierung und umwandlung in tensor
            norm = normalize(images)
            images = norm
        return (images,self.labels[idx])
    
    def set_mean_std(self,mean,std):
        self.mean = mean
        self.std = std
