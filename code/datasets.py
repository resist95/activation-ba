import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from data import caltech101
from data import CIFAR10
from data import MNIST
from context import train_test_validate_split
from context import validate_split

class caltech101dataset():

    def __init__(self,train_ratio, test_ratio, validation_ratio):
        self.cal = caltech101()
        self.data,self.labels = self.cal.get_data()
        self.X_train,self.X_test,self.X_val,self.y_train,self.y_test,self.y_val = \
            train_test_validate_split(train_ratio, test_ratio, validation_ratio,self.data,self.labels)
    
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
        self.X_val,self.y_val = validate_split(0.16,validation_ratio,self.X_test,self.y_test)
    
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
        self.X_val,self.y_val = validate_split(0.14,validation_ratio,self.X_test,self.y_test)

    def get_split_data(self):
        dic = {'X_train':self.X_train,
            'X_test':self.X_test,
            'X_val':self.X_val,
            'y_train':self.y_train,
            'y_test':self.y_test,
            'y_val':self.y_val}
        return dic


def main():
    cal = mnistdataset(0.7,0.2,0.1)



if __name__== "__main__":

    main()