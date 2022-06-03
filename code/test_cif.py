import sys,os
import torch
import numpy as np

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.data import CIFAR10
from datasets.datasets import CustomDataset
from models.cifar import CNN_CIFAR_RELU,CNN_CIFAR_SWISH,CNN_CIFAR_TANH
from models.cifar import CNN_CIFAR_RELU_drop_sched_0,CNN_CIFAR_RELU_drop_sched_1,CNN_CIFAR_RELU_drop_sched_2,CNN_CIFAR_RELU_drop_sched_3,CNN_CIFAR_RELU_drop_sched_4,CNN_CIFAR_RELU_drop_sched_5
from models.cifar import CNN_CIFAR_SWISH_drop_sched_0, CNN_CIFAR_SWISH_drop_sched_1, CNN_CIFAR_SWISH_drop_sched_2, CNN_CIFAR_SWISH_drop_sched_3
from models.cifar import CNN_CIFAR_TANH_drop_sched_0,CNN_CIFAR_TANH_drop_sched_1,CNN_CIFAR_TANH_drop_sched_2,CNN_CIFAR_TANH_drop_sched_3,CNN_CIFAR_TANH_drop_sched_4
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
            
    m_names = ['relu','swish','tanh']
    
    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')
    
    for i,model in enumerate(m):

        print(f'Training CNN with activation function [{m_names[i]}]')
        a = ActivationFunction(model,f'CIFAR_REFERENZ_TO_BEAT_{m_names[i]}',params_dict_cifar,m_names[i])
        a.compute_drop_sched(train,test,10)

def gradients():
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = params_dict_cifar['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = CIFAR10(0.0,'test')
    data.prepare_data_grad(200)
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
    m = [CNN_CIFAR_RELU(),CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['relu','swish', 'tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient',params_dict_cifar,m_names[i])
        a.compute_gradients_per_class(train,test,10)

def activations():
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
    m = [CNN_CIFAR_RELU_fill(),CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'act',params_dict_cifar,m_names[i])
        a.compute_activations(train,test)

def feature_map():
    
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = 1
    params_dict_cifar['max_epochs'] = 10
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
    
    m = [CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_cifar,m_names[i])
        a.compute_activations_feature_map(train,test)

def activations_input():
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
    m = [CNN_CIFAR_RELU_fill(),CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'act',params_dict_cifar,m_names[i])
        a.compute_activations_in(train,test)

def feature_map_per_layer():
    
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = 1
    params_dict_cifar['max_epochs'] = 5
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
    
    m = [CNN_CIFAR_RELU_fill(),CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_cifar,m_names[i])
        a.compute_activations_feature_map_per_layer(train,test)

def feature_map_input():
    
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = 1
    params_dict_cifar['max_epochs'] = 15
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
    
    m = [CNN_CIFAR_RELU_fill(),CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_cifar,m_names[i])
        a.compute_activations_feature_map_in(train,test)

def gradients_input():
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
        
        a = ActivationFunction(model,'gradient_input',params_dict_cifar,m_names[i])
        a.compute_gradients_input(train,test)

def gradients_output():
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
        
        a = ActivationFunction(model,'gradient_ouput',params_dict_cifar,m_names[i])
        a.get_gradients_hook(train,test)

def feature_map_grad():
    
    batch_size_train = params_dict_cifar['batch_size']
    batch_size_test = 1
    params_dict_cifar['max_epochs'] = 5
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
    
    m = [CNN_CIFAR_RELU_fill(),CNN_CIFAR_SWISH_fill(),CNN_CIFAR_TANH_fill()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_cifar,m_names[i])
        a.compute_gradients_feature_map_in(train,test)

def gradients_hook_classes():
    batch_size_train = 1
    batch_size_test = params_dict_cifar['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = CIFAR10(0.0,'test')
    data.prepare_data_grad(100)
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
        
        a = ActivationFunction(model,'gradient',params_dict_cifar,m_names[i])
        a.compute_gradients_per_class_hook(train,test,10,i)

def gradients_input_output():
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
        
        a = ActivationFunction(model,'gradient_input',params_dict_cifar,m_names[i])
        a.get_gradients_hook_in_out(train,test)


def gradients_input_output_all_layers():
    batch_size_train = 1
    batch_size_test = params_dict_cifar['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = CIFAR10(0.0,'test')

    data.prepare_data_grad(200)
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
        
        a = ActivationFunction(model,'gradient_input',params_dict_cifar,m_names[i])
        a.compute_gradients_per_class_hook_in_out_all(train,test)

def main():
    accuracy_loss(val=False)
    #gradients()
    #activations()
    #feature_map()
    #activations_input()
    #feature_map_input()
    #gradients_input()
    #feature_map_per_layer()
    #feature_map_grad()
    #gradients_output()
    #gradients_hook_classes()
    #gradients_input_output()
    #gradients_input_output_all_layers()
        
if __name__== "__main__":

    main()