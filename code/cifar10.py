from random import shuffle
from turtle import down
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self,labels,img,transform=None,target_transform=None):
        self.img_labels = labels
        self.images = img
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        data = self.images[idx][:]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label





    
    


def main():
    cif = caltech101('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/')

    #cifar_train_data,cifar_train_labels,cifar_test_data,cifar_test_labels = cif.load_data()
    #cif.print_data(cifar_train_data,cifar_train_labels)


if __name__== "__main__":

    main()