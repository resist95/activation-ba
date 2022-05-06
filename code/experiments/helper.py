#default dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

#special dependencies
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

sns.set()

def prepare_summary_writer(name,act_fn,**kwargs):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    x = current_time.replace(':','_')
    writer = SummaryWriter(f'runs/{name}_{act_fn}_{x}_{kwargs}')
    return writer

def prepare_device():
    device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda:0':
        print('running on gpu')
    else:
        print('running on cpu')
    return device

def plot_loss_acc(train_acc,test_acc,train_loss,test_loss,epochs,act_fn):
    i = np.min(train_acc) if np.min(train_acc) > np.min(test_acc) else np.min(test_acc)
    fig, axs = plt.subplots(2)
    axs[0].plot(train_acc,label='Train Accuracy')
    axs[0].plot(test_acc,label='Test Accuracy')
    axs[0].set_xticks(np.arange(0, epochs,1))
    axs[0].set_yticks(np.arange(0.4,1,0.2))
    axs[0].set_title(f'Accuracy for Activation function {act_fn}')
    axs[1].plot(train_loss,label='Train Loss')
    axs[1].plot(test_loss,label='Test Loss')
    axs[1].set_xticks(np.arange(0,epochs,1))
    axs[1].set_yticks(np.arange(0,1.6,0.2))
    axs[1].set_title(f'Loss for Activation function {act_fn}')
    
    axs[0].legend()
    axs[1].legend()
    plt.savefig(f'cifar_{act_fn}.jpg')
    plt.clf()

def plot(data,plot_dict,epochs=0):
    names = []
    for key in data.keys():
        names.append(key)
    
    fig, axs = plt.subplot(plot_dict['row'],plot_dict['col'])

    if plot_dict['type'] == 'hist':
        i = 0
        for row in range(plot_dict['row']):
            for col in range(plot_dict['col']):
                axs[row,col].hist(data=data[names[i]],
                                    bins=plot_dict['bins'],
                                    label=plot_dict['label'])
                i +=1
    elif plot_dict['type'] == 'line':
        i = 0
        for row in range(plot_dict['row']):
            for col in range(plot_dict['col']):
                axs[row,col].plot(data=(data[names[i]],plot_dict['epochs']),
                                    label=plot_dict['label'])
                i +=1
    #plt.legend()
    #plt.show()

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
    return 1

def get_activations(model,data):
    activation = {}
    model.eval()
    with torch.no_grad():
        for name, layer in model.named_children():
            if isinstance(layer, nn.Sequential):
                get_activations(layer)
            else:
                layer_name = name
                i = layer(data)
                activation[layer_name] = i.view(-1).cpu().detach().numpy()
    return activation


def get_gradients(model,p):
    grad = {}
    for name, layer in model.named_children():
        if isinstance(layer,nn.Sequential):
            get_gradients(layer,p)
        else:
            layer_name = name
            for name, param in layer.named_parameters():
                if isinstance(layer, nn.Linear):
                    if p in name:
                        grad[layer_name] = param.view(-1).cpu().detach().numpy()
                if isinstance(layer, nn.Conv2d):
                    if p in name:
                        grad[layer_name] = param.view(-1).cpu().detach().numpy()
    return grad

def get_layer_names(model):
    names = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            get_layer_names(layer)
        else:
            names.append(name)
    return names