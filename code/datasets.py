
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class Caltech101Dataset(Dataset):

    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224,224))
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if self.transform:
            images = self.transform(self.images[idx])
        return (images,self.labels[idx])


class IntelDataset(Dataset):

    def __init__(self,images,labels):
        self.images = np.array(images)
        self.labels = labels
        self.transform = T.Compose([
        T.ToTensor()
    ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if self.transform:
            images = self.transform(self.images[idx])
            images = torch.transpose(images,0,1)
        return (images,self.labels[idx])


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
        return (images,self.labels[idx])


class MnistDataset():

    def __init__(self, images,labels):
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
        return (images,self.labels[idx])


def main():
    i = CIFAR10()
    train_x,train_y,test_x,test_y = i.get_data()
    train_data = Cifar10Dataset(test_x,test_y)
    train_load = torch.utils.data.DataLoader(dataset=train_data,batch_size=32,shuffle=True)

    for x , y in train_load:
        print(y[0])

if __name__== "__main__":

    main()
