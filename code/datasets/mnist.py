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


class MNIST():

    def __init__(self,path_to_dir,download=False):
        self.mnist_train_data = datasets.MNIST(root=path_to_dir, train=True, download=download)
        self.mnist_test_data = datasets.MNIST(root=path_to_dir,train=False,download=download)
       