
from data import Intel
from data import CIFAR10
from data import MNIST

from context import validate_split
import torch
import torchvision
import torchvision.transforms as transforms


class inteldataset():

    def __init__(self,validation_ratio):
        self.int = Intel()
        self.data,self.labels = self.int.get_data()
        train_data = self.data[0:14034]
        train_labels = self.labels[0:14034]
        test_data = self.data[14034:]
        test_labels = self.labels[14034:]

        self.X_train = train_data
        self.y_train = train_labels
        self.X_test = test_data
        self.y_test = test_labels
        self.X_val,self.y_val = validate_split(0.2,validation_ratio,self.X_test,self.y_test)
     
    def get_split_data(self):
        dic = {'X_train':self.X_train,
            'X_test':self.X_test,
            'X_val':self.X_val,
            'y_train':self.y_train,
            'y_test':self.y_test,
            'y_val':self.y_val}
        return dic


class cifar10dataset():
    
    def __init__(self,validation_ratio):
        self.cif = CIFAR10()
        self.data,self.labels = self.cif.get_data()
        train_data = self.data[0:50000]
        train_labels = self.labels[0:50000]
        test_data = self.data[50000:]
        test_labels = self.labels[50000:]
        
        self.X_train = train_data
        self.y_train = train_labels
        self.X_test = test_data
        self.y_test = test_labels
        self.X_val,self.y_val = validate_split(0.2,validation_ratio,self.X_test,self.y_test)
    
    def get_transforms(self):
        mean,std = self.cif.get_mean_std_deviation()
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((mean),(std))])
        return transform
    
    def get_split_data(self):
        dic = {'X_train':self.X_train,
            'X_test':self.X_test,
            'X_val':self.X_val,
            'y_train':self.y_train,
            'y_test':self.y_test,
            'y_val':self.y_val}
        return dic


class mnistdataset():

    def __init__(self, validation_ratio):
        self.mnist = MNIST()
        self.data,self.labels = self.mnist.get_data()
        train_data = self.data[0:60000]
        train_labels = self.labels[0:60000]
        test_data = self.data[60000:]
        test_labels = self.labels[60000:]

        self.X_train = train_data
        self.y_train = train_labels
        self.X_test = test_data
        self.y_test = test_labels
        self.X_val,self.y_val = validate_split(0.2,validation_ratio,self.X_test,self.y_test)
    
    def get_transforms(self):
        mean,std = self.mnist.get_mean_std_deviation()
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((mean),(std))])
        return transform
    
    def get_split_data(self):
        dic = {'X_train':self.X_train,
            'X_test':self.X_test,
            'X_val':self.X_val,
            'y_train':self.y_train,
            'y_test':self.y_test,
            'y_val':self.y_val}
        return dic

def main():
    cal = mnistdataset(0.1)
    transform = cal.get_transforms()
    print(transform)

if __name__== "__main__":

    main()