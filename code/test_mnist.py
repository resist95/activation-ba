import sys,os
import torch
import numpy as np

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.data import MNIST
from datasets.datasets import CustomDataset
from models.mnist import CNN_MNIST_RELU,CNN_MNIST_SWISH,CNN_MNIST_TANH
from models.mnist import CNN_MNIST_RELU_drop_sched_0,CNN_MNIST_SWISH_drop_sched_0,CNN_MNIST_TANH_drop_sched_0
from experiments.parameters import params_dict_mnist
from experiments.activation import ActivationFunction
from experiments.activation import plot_single

from torch.utils.tensorboard import SummaryWriter

def accuracy_loss(val=True):
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    print('Loading Data... \n')
    if val == True:
        data = MNIST(0.1,'validate')
    else:
        data = MNIST(0.0,'test')
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

    m = [CNN_MNIST_RELU_drop_sched_0(),CNN_MNIST_SWISH_drop_sched_0(),CNN_MNIST_TANH_drop_sched_0()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')
    for i,model in enumerate(m):
        n = m_names[i]
        print(f'Training CNN with activation function [{m_names[i]}]')
        a = ActivationFunction(model,'MNIST_DROP_SCHED_log_i_exp1_5_{n}',params_dict_mnist,m_names[i])
        a.compute_drop_sched(train,test,15)

def accuracy_loss_batch():
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    print('Loading Data... \n')

    data = MNIST(0.0,'test')
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

    

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')
    gamma_range = np.arange(0.152,0.175,0.001)
    for g in range(len(gamma_range)):
        print('gamma',gamma_range[g])
        gg = gamma_range[g]
        for i in range(1):
            m = [CNN_MNIST_RELU_drop_sched_0()]
            m_names = ['relu']
            n = m_names[i]
            print(f'Training CNN with activation function [{m_names[i]}]')
            a = ActivationFunction(m[i],f'MNIST_DROP_drop_log_var8_{gg}_{n}',params_dict_mnist,m_names[i])
            a.compute_drop_sched_batch(train,test,50,'drop_log_var8',gamma_range[g]) 
def gradients():
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    acc = {}
    loss = {}
    #morgen tanh
    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient',params_dict_mnist,m_names[i],10,'act_comp')
        a.compute_gradients_per_class(train,test)

def activations():
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'act',params_dict_mnist,m_names[i],10,'acti')
        a.compute_activations(train,test)

def feature_map():
    
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = 1
    params_dict_mnist['max_epochs'] = 15
    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_mnist,m_names[i])
        a.compute_activations_feature_map(train,test)

def activations_input():
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'act',params_dict_mnist,m_names[i])
        a.compute_activations_in(train,test)

def feature_map_input():
    
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = 1
    params_dict_mnist['max_epochs'] = 15
    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_mnist,m_names[i])
        a.compute_activations_feature_map_in(train,test)

def gradients_input():
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient_input',params_dict_mnist,m_names[i])
        a.compute_gradients_input(train,test)

def feature_map_act_per_layer():
    
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = 1
    params_dict_mnist['max_epochs'] = 5
    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    
    m = [CNN_MNIST_TANH()]
    m_names = ['tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_mnist,m_names[i])
        a.compute_activations_feature_map_per_layer(train,test)

def gradients_input_map():
    batch_size_train = params_dict_mnist['batch_size']
    batch_size_test = params_dict_mnist['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = MNIST(0.0,'test')
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
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient_input',params_dict_mnist,m_names[i])
        a.compute_gradients_feature_map_in(train,test)

def gradients_input_output_all_layers():
    batch_size_train = 1
    batch_size_test = params_dict_mnist['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = MNIST(0.0,'test')

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
    m = [CNN_MNIST_RELU(),CNN_MNIST_SWISH(),CNN_MNIST_TANH()]
    m_names = ['relu','swish', 'tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient_input',params_dict_mnist,m_names[i])
        a.compute_gradients_per_class_hook_in_out_all(train,test)

def main():
    #accuracy_loss(val=False)
    accuracy_loss_batch()
    #feature_map()
    #gradients()
    #activations()
    #activations_input()
    #feature_map_input()
    #gradients_input()
    #feature_map_act_per_layer()
    #gradients_input_map()
    #gradients_input_output_all_layers()
    
        
if __name__== "__main__":


    main()