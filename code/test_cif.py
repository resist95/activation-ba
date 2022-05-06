import numpy as np
import matplotlib.pyplot as plt
from pandas import lreshape
import seaborn as sns
from sklearn.utils import shuffle
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math
import sys
import os
from collections import OrderedDict
from typing import Dict, Callable
import torch

from models.cifar_models import CNN_CIFAR_RELU, CNN_CIFAR_RELU_2, CNN_CIFAR_RELU_3
from models.cifar_models import CNN_CIFAR_SWISH
from models.cifar_models import CNN_CIFAR_TANH,CNN_CIFAR_TANH_2,CNN_CIFAR_TANH_3
from datasets.datasets import CustomDataset 
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.data import Intel

from datasets.data import MNIST
from models.mnist_models import CNN_MNIST_ACT


from datasets.data import CIFAR10

sns.set()

import torch

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
x = current_time.replace(':','_')


#device setup 
device =  'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('running on gpu')

writer = SummaryWriter()
#Params 0.1, 0.03, 0.001, 0.0003
epochs = 20
batch_size = 64
lr = 0.0005

#Data
dat = CIFAR10(0.1,'validate')
dat.prepare_data()
m,s = dat.get_mean_std()
#dataset
ds = CustomDataset
ds.set_mean_std(ds,mean=m,std=s)

criterion = nn.CrossEntropyLoss()
#get data for dl
X_train,y_train = dat.get_data('train')
X_val,y_val = dat.get_data('val')
train = ds(X_train,y_train)
test = ds(X_val,y_val)

trains = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
tests = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)


def evaluate(X, y, train=False):
    if train:
        model.zero_grad()
    scores = model(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,y)]
    acc = matches.count(True)/len(matches)

    loss = criterion(scores,y)
    if train:
        loss.backward()
        optimizer.step()
    return acc,loss.item()

def train(epoch,train):
    model.train()
    acc = 0.0
    loss = 0.0
    for idx,(data,targets) in enumerate(train):
        
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        accs, losses = evaluate(data,targets,train=True)
        loss += losses * data.size(0)
        acc += accs * data.size(0)
    
    mean_acc = acc / len(train.dataset)
    mean_loss = loss / len(train.dataset)
    writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
    writer.add_scalar('Mean Loss Train',mean_loss,epoch)
    return mean_acc, mean_loss
def test(epoch,test):
    model.eval()
    acc = 0.0
    loss = 0.0
    with torch.no_grad():
        for (data,targets) in test:
            data = data.to(device=device)
            targets = targets.to(device=device)

            accs, losses = evaluate(data,targets,train=False)
            loss += losses * data.size(0)
            acc += accs * data.size(0)
    mean_acc = acc / len(test.dataset)
    mean_loss = loss / len(test.dataset)
    writer.add_scalar('Mean Accuracy Test',mean_acc,epoch)
    writer.add_scalar('Mean Loss Test',mean_loss,epoch)
    return mean_acc, mean_loss
#["0.000125", "0.00025", "0.0005"]
weight = 0.000125
m = [CNN_CIFAR_TANH(),CNN_CIFAR_TANH_2(),CNN_CIFAR_TANH_3()]
for i,model in enumerate(m):
    model = CNN_CIFAR_RELU()
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight)
    model.to(device)
    writer = SummaryWriter(f'runs/cifar_relu_{i}_lr{lr}_bn1-4_{x}')    
    for epoch in range(epochs):
        print(f'Epoch: [{epoch+1} / {epochs}] [{lr}] [{weight}]')
        train_acc,train_loss = train(epoch,trains)
        test_acc,test_loss = test(epoch,tests)
        diff = train_acc - test_acc
        print(f'Test loss: {test_loss} Train loss: {train_loss} Difference: {diff}')
        diff = 0