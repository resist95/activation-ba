
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
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
        self.soft = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        
        if self.act_fn_name == 'tanh':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d)):
                    nn.init.constant_(m.weight,1)
                    nn.init.constant_(m.bias, 0)
        else:            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def _make_layers(self,act_fn):
        model = nn.Sequential(
            nn.Conv2d(1,8,3,1),
            act_fn(),
            nn.BatchNorm2d(8),
            
            
            nn.MaxPool2d(2,2),

            nn.Conv2d(8,16,3,1,1),
            act_fn(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.25),
            
        )
        return model
    
    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        if(self.act_fn_name == 'relu'):
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.soft(out)
            return out
        elif(self.act_fn_name == 'tanh'):
            out = F.tanh(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.soft(out)
            return out
        elif(self.act_fn_name == 'swish'):
            out = F.silu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.soft(out)
            return out
        elif(self.act_fn_name == 'leakyrelu'):
            out = F.leaky_relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.soft(out)
            return out
        elif(self.act_fn_name == 'gelu'):
            out = F.gelu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.soft(out)            
            return out

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
def print_layers(net,param,direction):
    #print(gradient)
    global step_conv
    global step_linear
    if param == 'activation':
        for name, layer in net.named_children():
            if isinstance(layer,nn.Sequential):
                print_layers(layer,param,direction)
            else:
               layer.register_forward_hook(hook_activation)
    elif param == 'gradient':
        for name, layer in net.named_children():
            if isinstance(layer,nn.Sequential):
                print_layers(layer,param,direction)
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
    
    def plots(self,n_epochs,train,test,param,direction):
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

                data,targets = next(iter(test))
                print_layers(self.model,param,direction)
                with torch.no_grad():

                    data.to(device)
                    targets.to(device)
                    scores = self.model(data)

                rows = len(activation)
                columns = 1
                data = []
                names = []

                for key in activation.keys():
                    data.append(activation[key])
                    names.append(key)

                fig, ax = plt.subplots(rows,columns, figsize=(columns*10,rows*2.5),squeeze=False)
                lv = 0
                for i in range(rows):
                    for l in range(columns):
                        ax[i][l].hist(x=data[lv], bins=100, color='C0', density=True)
                        ax[i][l].set_title(label = f"Layer {names[lv]}")
                    lv+=1
        
                fig.suptitle(f"Activation distribution for activation function {act}", fontsize=14)
                fig.subplots_adjust(hspace=0.4, wspace=0.4)
                plt.savefig(f'activation_{act}_{epoch}.png')
                plt.show()
                clear_everything(fig)
    
    def plot_gradient(self,n_epochs,train,param,direction):
        global step
        for act in self.act_fn_name:
            for epoch in range(n_epochs):
                print(f'Epoch: {epoch} Act: {act}')
                self.model = CNN_MNIST_ACT(self.act_fn_by_name[act],act)
                self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)
                for idx,(img,targets) in enumerate(train):
                    self.model.to(device)
                    self.model.zero_grad()

                    img.to(device)
                    targets.to(device)

                    scores = self.model(img)
                    
                    loss = self.criterion(scores,targets)
                    loss.backward()
                print_layers(self.model,param,direction)
                rows = len(gradient)
                columns = 1
                data = []
                names = []

                for key in gradient.keys():
                    data.append(gradient[key])
                    names.append(key)

                fig, ax = plt.subplots(rows,columns, figsize=(columns*10,rows*2.5),squeeze=False)
                lv = 0
                for i in range(rows):
                    for l in range(columns):
                        ax[i][l].hist(x=data[lv], bins=100, color='C0', density=True)
                        ax[i][l].set_title(label = f"Layer {names[lv]}")
                    lv+=1
        
                fig.suptitle(f"Gradient distribution for activation function {act}", fontsize=14)
                fig.subplots_adjust(hspace=0.4, wspace=0.4)
                plt.savefig(f'gradientplot_{act}_{epoch}.png')
                
                clear_everything(fig)

def clear_everything(fig):
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
    fig.clf()
    plt.close()
    activation.clear()
    gradient.clear()

def main():

    #define param here
    batch_size = 128
    n_epochs = 30
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
    #act.plot_gradient(30,train,'gradient','backward')
    act.plots(n_epochs,train,test,'activation','forward')
        
if __name__== "__main__":

    main()