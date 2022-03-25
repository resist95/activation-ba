


from data import CIFAR10
from data import caltech101
from data import MNIST
from data import Intel

from datasets import Caltech101Dataset
from datasets import Cifar10Dataset
from datasets import MnistDataset
from datasets import IntelDataset

from cnn import CNN_CIFAR

import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchviz 


def make_train_step(model,loss_fn,optimizer):
    def train_step(X,y):
        model.train()        
        pred = model(X)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step

#device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#define param here
val_ratio = 0.1
batch_size = 32
n_epochs = 50
lr = 0.01


#load data
print('Loading Data... \n')
cif = CIFAR10()
cal = caltech101()
mni = MNIST()
inte = Intel()

data = [cif,mni,inte,cal]
data_names = ['MNIST_Dataset','CIFAR_Dataset','Intel_Dataset','Caltech_Dataset']

print('Done.')

#load into dataloader
mnis = MnistDataset
cifa = Cifar10Dataset
calt = Caltech101Dataset
intel = IntelDataset

datasets = [cifa,mnis,intel,calt]
train_loader = []
test_loader = []

print('Loading train and test samples into DataLoader... \n')
for i,dat in enumerate(data):
    X_train,y_train,X_test,y_test = dat.get_data()
    train = datasets[i](X_train,y_train)
    test = datasets[i](X_test,y_test)
    train_l = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    test_l = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
    train_loader.append(train_l)
    test_loader.append(test_l)

print('Done')

#declare nn here
mni_nn = 'define here'
cif_nn = CNN_CIFAR()
int_nn = 'define here'
cal_nn = 'define here'

model = [mni_nn,cif_nn,int_nn,cal_nn]

losses = []
val_losses = []

i = 0
for train,test in zip(train_loader,test_loader):
    #define loss,optimizer
    curr_model = cif_nn
    curr_model = curr_model.to(device)    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(curr_model.parameters(),lr=lr)    
    train_step = make_train_step(curr_model,loss_fn,optimizer)
    
    for epoch in range(n_epochs):
        for X_train,y_train in train:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            loss = train_step(X_train,y_train)
            losses.append(loss)
        
        with torch.no_grad():
            for X_test,y_test in test:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                
                curr_model.eval()
                x_val = curr_model(X_test)
                val_loss = loss_fn(x_val,y_test)
                val_losses.append(val_loss.item())
    
    #PATH = '../models/'
    #torch.save(model[num].state_dict(), PATH)
    i +=1
          


