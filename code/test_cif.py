import sys,os
import torch
import numpy as np

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.data import CIFAR10
from datasets.datasets import CustomDataset
from models.cifar import CNN_CIFAR_RELU,CNN_CIFAR_SWISH,CNN_CIFAR_TANH

from experiments.parameters import params_dict_cifar
from experiments.activation import ActivationFunction

from torch.utils.tensorboard import SummaryWriter

def accuracy_loss(val=True):
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = params_dict_cifar['batch_size']

    print('Loading Data... \n')
    if val == True:
        data = CIFAR10(0.1,'validate')
    else:
        data = CIFAR10(0.0,'test')
    data.prepare_data()
    m,s = data.get_mean_std()
    print('Done.')

    dataset = CustomDataset

    print('Loading train and test samples into DataLoader... \n')
    X_train,y_train = data.get_data('train')
    if val == True:
        X_test, y_test = data.get_data('val')
    else: 
        X_test, y_test = data.get_data('test')
    
    train = dataset(X_train,y_train)
    test = dataset(X_test,y_test)
    train.set_mean_std(m,s)
    test.set_mean_std(m,s)
    train = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size_train,shuffle=False)
    test = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size_test,shuffle=False)
    print('Done')

    m = [CNN_CIFAR_RELU(),CNN_CIFAR_SWISH(),CNN_CIFAR_TANH()]
    m_names = ['relu', 'swish', 'tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')
    
    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        a = ActivationFunction(model,'',params_dict_cifar,m_names[i],10,'act_comp')
        a.compute(train,test,100)

def gradients():
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = params_dict_cifar['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = CIFAR10(0.0,'test')
    data.prepare_data()
    m,s = data.get_mean_std()
    print('Done.')

    dataset = CustomDataset

    print('Loading train and test samples into DataLoader... \n')
    X_train,y_train = data.get_data('train')
    X_test, y_test = data.get_data('test')
    
    train = dataset(X_train,y_train)
    test = dataset(X_test,y_test)
    train.set_mean_std(m,s)
    test.set_mean_std(m,s)
    train = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size_train,shuffle=False)
    test = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size_test,shuffle=False)
    print('Done')
    m = [CNN_CIFAR_RELU(),CNN_CIFAR_SWISH(),CNN_CIFAR_TANH()]
    m_names = ['relu','swish', 'tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient',params_dict_cifar,m_names[i],10,'act_comp')
        a.compute_gradients(train,test)

def activations():
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = params_dict_cifar['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = CIFAR10(0.0,'test')
    data.prepare_data_act(200)
    m,s = data.get_mean_std()
    print('Done.')

    dataset = CustomDataset

    print('Loading train and test samples into DataLoader... \n')
    X_train,y_train = data.get_data('train')
    X_test, y_test = data.get_data('test')
    
    train = dataset(X_train,y_train)
    test = dataset(X_test,y_test)
    train.set_mean_std(m,s)
    test.set_mean_std(m,s)
    train = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size_train,shuffle=False)
    test = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size_test,shuffle=False)
    print('Done')
    m = [CNN_CIFAR_RELU(),CNN_CIFAR_SWISH(),CNN_CIFAR_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'act',params_dict_cifar,m_names[i],10,'act_comp')
        a.compute_activations(train,test)




def main():
    #accuracy_loss(val=False)
    #gradients()
    activations()
    
        
if __name__== "__main__":

    main()