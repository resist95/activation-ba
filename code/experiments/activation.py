
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.datasets import MnistDataset
from datasets.data import MNIST

sns.set()

import torch

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
x = current_time.replace(':','_')


#device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CNN_MNIST_ACT(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_MNIST_ACT,self).__init__()
        self.act_fn = act_fn
        self.act_fn_name = name
        self.layers = self._make_layers(self.act_fn)

        self.fc1 = nn.Linear(2704,128)
        self.fc2 = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.25)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self,act_fn):
        model = nn.Sequential(
            nn.Conv2d(1,8,3,1),
            nn.BatchNorm2d(8),
            act_fn(),
            
            nn.MaxPool2d(2,2),

            nn.Conv2d(8,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.25),
            act_fn(),
        )
        return model
    
    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        if(self.act_fn_name == 'relu'):
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'tanh'):
            out = F.tanh(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'swish'):
            out = F.silu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'leakyrelu'):
            out = F.leaky_relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'gelu'):
            out = F.gelu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out

activation = {}
step = 1
def hook_activation(model, input,output):
    global step
    if isinstance(model,nn.Conv2d):
        activation[f'conv2d_{step}'] = output.view(-1).detach().numpy()
    if isinstance(model,nn.Linear):
        activation[f'linear_{step}'] = output.view(-1).detach().numpy()
    if isinstance(model,nn.MaxPool2d):
        activation[f'maxpool_{step}'] = output.view(-1).detach().numpy()
    if isinstance(model,nn.BatchNorm2d):
        activation[f'batchnorm_{step}'] = output.view(-1).detach().numpy()   

gradient = {}
def hook_gradient(model, input,output):
    global step
    #gradient = {name: data.view(-1).detach().numpy for name, data in model.named_parameters() if 'weight' in name}
    m = model.named_parameters()
    print(m)

def print_layers(net,param):
    for name, layer in net.named_children():
        if isinstance(layer,nn.Sequential):
            print_layers(layer,param)
        else:
            if param == 'activation':
                layer.register_forward_hook(hook_activation)
            elif param == 'gradient':
                layer.register_forward_hook(hook_gradient)



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
            self.model = CNN_MNIST_ACT(self.act_fn_by_name[act],act)
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)
            self.model.to(device)
            self.writer = SummaryWriter(f'runs/mnist_activation_{act}_{x}')
            
            for epoch in range(n_epochs):
                print(f'Epoch: [{epoch+1} / {n_epochs}] current activation function: {act}')
                self.train(epoch,train)

                self.test(epoch,test)
    
    def plots(self,n_epochs,train,test,param):
        global step

        for act in self.act_fn_name:
            self.model = CNN_MNIST_ACT(self.act_fn_by_name[act],act)
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)
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

            #state_dict = self.model.state_dict()
            #orch.save(state_dict, f'/trained_models/mnist_model_{act}')

            data,targets = next(iter(test))
            with torch.no_grad():
                data.to(device)
                targets.to(device)
                print_layers(self.model,param)

                scores = self.model(data)

            if param == 'activation':
                rows = len(activation)
                columns = 1
                data = []
                for key in activation.keys():
                    data.append(activation[key])
            
            elif param == 'gradient':
                rows = len(gradient)
                columns = 1
                data = []
                for key in gradient.keys():
                    data.append(gradient[key])

            fig, ax = plt.subplots(rows,columns, figsize=(columns*2.7,rows*2.5),squeeze=False)
            lv = 0
            for i in range(columns):
                for l in range(rows):
                    ax[l][i].hist(x=data[lv], bins=50, color='C0', density=True)
                    ax[l][i].set_title(label = f"Layer {key} - {self.model.layers[i].__class__.__name__}")
                    lv+=1
            
            if param == 'activation':
                fig.suptitle(f"Activation distribution for activation function {act}", fontsize=14)
                fig.subplots_adjust(hspace=0.4, wspace=0.4)
                plt.savefig(f'/plots/activationplot_{act}.png')
            elif param == 'gradient':
                fig.suptitle(f"Gradient distribution for activation function {act}", fontsize=14)
                fig.subplots_adjust(hspace=0.4, wspace=0.4)
                plt.savefig(f'/plots/gradientplot_{act}.png')
            plt.clf()
            plt.close()
            data.clear()
            activation.clear()
            gradient.clear()

def main():

    #define param here
    batch_size = 128
    n_epochs = 1
    lr = 0.001

    #load data
    print('Loading Data... \n')
    data = MNIST()

    print('Done.')

    #load into dataloader
    dataset = MnistDataset

    print('Loading train and test samples into DataLoader... \n')
    X_train,y_train,X_test,y_test = data.get_data()
    train = dataset(X_train,y_train)
    test = dataset(X_test,y_test)
    train = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    test = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
    
    #load activation function class
    writer = SummaryWriter
    act = ActivationFunction(lr,writer)
    #act.compare_activation_functions(2,train,test)
    act.plots(n_epochs,train,test,'gradient')
        
if __name__== "__main__":

    main()