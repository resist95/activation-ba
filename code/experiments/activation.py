
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math
import sys
import os
from collections import OrderedDict
from typing import Dict, Callable

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.datasets import IntelDataset
from datasets.data import Intel
from models.intel_models import CNN_INTEL

from datasets.datasets import MnistDataset
from datasets.data import MNIST
from models.mnist_models import CNN_MNIST_ACT


from datasets.datasets import Cifar10Dataset
from datasets.data import CIFAR10
from models.cifar_models import CNN_CIFAR_COLAB
sns.set()

import torch

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
x = current_time.replace(':','_')


#device setup 
device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
if device == 'cuda:0':
    print('running on gpu')

activation = {}
step_conv = 1
step_linear = 1
step_batch = 1
step_pool = 1
step_activation = 1
def hook_activation(model, input,output):
    global step_conv
    global step_linear
    global step_batch
    global step_pool
    global step_activation
    if isinstance(model,nn.Conv2d):
        activation[f'conv2d_{step_conv}'] = output.view(-1).detach().numpy()
        step_conv+=1
    if isinstance(model,nn.Linear):
        activation[f'linear_{step_linear}'] = output.view(-1).detach().numpy()
        step_linear +=1
    if isinstance(model,nn.MaxPool2d):
        activation[f'maxpool_{step_pool}'] = output.view(-1).detach().numpy()
        step_pool += 1
    if isinstance(model,nn.BatchNorm2d):
        activation[f'batchnorm_{step_batch}'] = output.view(-1).detach().numpy()
        step_batch += 1
    if isinstance(model,nn.Tanh):
        activation[f'tanh_{step_activation}'] = output.view(-1).detach().numpy()
        step_activation += 1
    if isinstance(model,nn.ReLU):
        activation[f'relu_{step_activation}'] = output.view(-1).detach().numpy()
        step_activation += 1
    if isinstance(model,nn.SiLU):
        activation[f'swish_{step_activation}'] = output.view(-1).detach().numpy()
        step_activation += 1
    if isinstance(model,nn.GELU):
        activation[f'gelu_{step_activation}'] = output.view(-1).detach().numpy()
        step_activation += 1
    if isinstance(model,nn.LeakyReLU):
        activation[f'leakyrelu_{step_activation}'] = output.view(-1).detach().numpy()
        step_activation += 1
    if isinstance(model,nn.Softmax):
        activation[f'softmax_{step_activation}'] = output.view(-1).detach().numpy()

gradient = {}
activation = {}
def print_layers(net,param):
    global step_conv
    global step_linear
    if param == 'activation':
        for name, layer in net.named_children():
            if isinstance(layer,nn.Sequential):
                print_layers(layer,param)
            else:
               layer.register_forward_hook(hook_activation)

    elif param == 'gradient':
        for name, layer in net.named_children():
            if isinstance(layer,nn.Sequential):
                print_layers(layer,param)
            else:
                for name, params in layer.named_parameters(): 
                    if isinstance (layer, nn.Linear):
                        if "weight" in name:
                            gradient[f'linear_{step_linear}'] = params.grad.view(-1).detach().numpy() 
                            step_linear += 1
                    if isinstance (layer, nn.Conv2d):
                        if 'weight' in name:
                            gradient[f'conv_{step_conv}'] = params.grad.view(-1).detach().numpy()
                            step_conv += 1

def remove_all_forward_hooks(model):
    for name,layer in model.named_children():
        for n, child in layer._modules.items():
            if child is not None:
                if hasattr(child,"_forward_hooks"):
                    child._forward_hooks  = OrderedDict()
                remove_all_forward_hooks(child)

class ActivationFunction:
    def __init__(self,lr,writer):
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.writer = writer
        self.lr = lr
        self.act_fn_name = ['tanh','relu','swish','leakyrelu','gelu']
        self.act_fn_by_name = {
                        "tanh": nn.Tanh,
                        "relu": nn.ReLU,
                        "swish": nn.SiLU,
                        "leakyrelu": nn.LeakyReLU,
                        "gelu": nn.GELU
                    }
    
    def evaluate(self,X, y, train=False):
        if train:
            self.model.zero_grad()
        else:
            self.model.eval()

        scores = self.model(X)
        matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,y)]
        acc = matches.count(True)/len(matches)

        loss = self.criterion(scores,y)
        if train:
            loss.backward()
            self.optimizer.step()
        return acc,loss.item()

    def train(self,epoch,train):
        self.model.train()
        acc = 0.0
        loss = 0.0
        for idx,(data,targets) in enumerate(train):
            
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            accs, losses = self.evaluate(data,targets,train=True)
            loss += losses * data.size(0)
            acc += accs * data.size(0)
        
        mean_acc = acc / len(train.dataset)
        mean_loss = loss / len(train.dataset)
        self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
        self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)
    
    def test(self,epoch,test):
        acc = 0.0
        loss = 0.0
        with torch.no_grad():
            for (data,targets) in test:
                data = data.to(device=device)
                targets = targets.to(device=device)

                accs, losses = self.evaluate(data,targets,train=False)
                loss += losses * data.size(0)
                acc += accs * data.size(0)
        mean_acc = acc / len(test.dataset)
        mean_loss = loss / len(test.dataset)
        self.writer.add_scalar('Mean Accuracy Test',mean_acc,epoch)
        self.writer.add_scalar('Mean Loss Test',mean_loss,epoch)

    def compare_activation_functions(self,n_epochs,train,test):
        for i, act in enumerate(self.act_fn_name):
            self.model = CNN_CIFAR_COLAB(self.act_fn_by_name[act],act)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
            self.model.to(device)
            self.writer = SummaryWriter(f'runs/intel_activation_{act}_{x}')
            
            for epoch in range(n_epochs):
                print(f'Epoch: [{epoch+1} / {n_epochs}] current activation function: {act}')
                self.train(epoch,train)

                self.test(epoch,test)
    
    def plots(self,n_epochs,train,test,param):
        for act in self.act_fn_name:
            self.model = CNN_MNIST_ACT(self.act_fn_by_name[act],act)
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
            self.model.to(device)

            for epoch in range( n_epochs):
                print(f'Epoch: [{epoch+1} / {n_epochs}] current activation function: {act}')
                self.model.train()
                for idx,(data,targets) in enumerate(train):

                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    
                    self.model.zero_grad()
                    scores = self.model(data)

                    loss = self.criterion(scores,targets)
                    loss.backward()
                    self.optimizer.step()
                if param == 'gradient':
                    print_layers(self.model,param)
                data,targets = next(iter(test))
                with torch.no_grad():
                    if param == 'activation':
                        print_layers(self.model,param)
                    data.to(device)
                    targets.to(device)
                    scores = self.model(data)
                data = []
                names = []
                if param == 'activation':
                    rows = len(activation)
                    for key in activation.keys():
                        data.append(activation[key])
                        names.append(key)
                else:
                    rows = len(gradient)
                    for key in gradient.keys():
                        data.append(gradient[key])
                        names.append(key)
                columns = 1            
                fig, ax = plt.subplots(rows,columns, figsize=(columns*10,rows*2.5),squeeze=False)
                lv = 0
                for i in range(rows):
                    for l in range(columns):
                        ax[i][l].hist(x=data[lv], bins='auto', color='C0', density=True)
                        ax[i][l].set_title(label = f"Layer {names[lv]}")
                    lv+=1

                if param == 'activation':
                    fig.suptitle(f"Activation distribution for activation function {act}", fontsize=14)
                    fig.subplots_adjust(hspace=0.4, wspace=0.4)
                    plt.savefig(f'activation_{act}.png')
                elif param == 'gradient':
                    fig.suptitle(f"Gradient distribution for activation function {act}", fontsize=14)
                    fig.subplots_adjust(hspace=0.4, wspace=0.4)
                    plt.savefig(f'gradientplot_{act}_{epoch}.png')
                remove_all_forward_hooks(self.model)
                clear_everything()
    
    
def clear_everything():
    global step_conv
    global step_linear
    global step_batch
    global step_pool
    global step_activation
    step_conv = 1
    step_linear = 1
    step_batch = 1
    step_pool = 1
    step_activation = 1
    plt.clf()
    plt.cla()
    plt.close()
    activation.clear()
    gradient.clear()

def main():

    #define param here
    batch_size = 256
    n_epochs = 20
    lr = 0.0001

    #load data
    print('Loading Data... \n')
    data = Intel()

    print('Done.')

    #load into dataloader
    dataset = IntelDataset

    print('Loading train and test samples into DataLoader... \n')
    X_train,y_train,X_test,y_test = data.get_data()
    train = dataset(X_train,y_train)
    test = dataset(X_test,y_test)
    train = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    test = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
    
    #load activation function class
    writer = SummaryWriter
    act = ActivationFunction(lr,writer)
    act.compare_activation_functions(n_epochs,train,test)
    #act.plots(n_epochs,train,test,'gradient')
        
if __name__== "__main__":

    main()