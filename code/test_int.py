import sys,os
import torch
import numpy as np

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from datasets.data import Intel
from datasets.datasets import CustomDataset
from models.intel import CNN_INTEL_RELU,CNN_INTEL_SWISH,CNN_INTEL_TANH
from models.intel import CNN_INTEL_RELU_drop_sched_0,CNN_INTEL_SWISH_drop_sched_0,CNN_INTEL_TANH_drop_sched_0

from experiments.parameters import params_dict_intel
from experiments.activation import ActivationFunction

from torch.utils.tensorboard import SummaryWriter


from models.intel import CNN_INTEL_TANH_drop_sched_0
from models.intel import CNN_INTEL_RELU_drop_sched_0
from models.intel import CNN_INTEL_SWISH_drop_sched_0
def accuracy_loss(val=True):
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']

    print('Loading Data... \n')
    if val == True:
        data = Intel(0.1,'validate')
    else:
        data = Intel(0.0,'test')
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
    m = [CNN_INTEL_RELU_drop_sched_0(),CNN_INTEL_SWISH_drop_sched_0(),CNN_INTEL_TANH_drop_sched_0()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i,model in enumerate(m):

        print(f'Training CNN with activation function [{m_names[i]}]')
        a = ActivationFunction(model,f'INTEL_DROP_SCHED_logi+3_6_{m_names[i]}',params_dict_intel,m_names[i])
        a.compute_drop_sched(train,test,15)

def accuracy_loss_batch():
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']

    print('Loading Data... \n')

    data = Intel(0.0,'test')
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
    m = [CNN_INTEL_RELU_drop_sched_0()]
    m_names = ['relu']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i,model in enumerate(m):

        print(f'Training CNN with activation function [{m_names[i]}]')
        a = ActivationFunction(model,f'INTEL_DROP_drop_log_var7_g0.7{m_names[i]}',params_dict_intel,m_names[i])
        a.compute_drop_sched_batch(train,test,15,'drop_log_var7')

def accuracy_loss_sched(val=True):
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']

    print('Loading Data... \n')
    if val == True:
        data = Intel(0.1,'validate')
    else:
        data = Intel(0.0,'test')
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
    m = [CNN_INTEL_RELU(),CNN_INTEL_RELU_drop_sched()]
    m_names = 'relu'

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i,model in enumerate(m):

        if i == 0:
            print(f'Training CNN with activation function [{m_names}]')
            a = ActivationFunction(model,'',params_dict_intel,'relu',6,i)
            a.compute(train,test,7)
        else:
            print(f'Training CNN with activation function [{m_names}]')
            a = ActivationFunction(model,'',params_dict_intel,'relu',6,i)
            a.compute_drop_sched(train,test,7)

def gradients():
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = Intel(0.0,'test')
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

    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient',params_dict_intel,m_names[i],6,'act_comp')
        a.compute_gradients_per_class(train,test)

def activations():
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']
    print(batch_size_test)
    print(batch_size_train)
    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = Intel(0.0,'test')
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
    
    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')
    
    for i,model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
    
        a = ActivationFunction(model,'act',params_dict_intel,m_names[i],10,'leaky')
        a.compute_activations(train,test)

def feature_map():
    
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = 1
    params_dict_intel['max_epochs'] = 15
    print('Loading Data... \n')
    data = Intel(0.0,'test')
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
    
    m = [CNN_INTEL_TANH()]
    m_names = ['tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_intel,m_names[i])
        a.compute_activations_feature_map(train,test)

def activations_input():
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']
    print(batch_size_test)
    print(batch_size_train)
    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = Intel(0.0,'test')
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
    
    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')
    
    for i,model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
    
        a = ActivationFunction(model,'act',params_dict_intel,m_names[i])
        a.compute_activations_in(train,test)

def feature_map_input():
    
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = 1
    params_dict_intel['max_epochs'] = 5
    print('Loading Data... \n')
    data = Intel(0.0,'test')
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
    
    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_intel,m_names[i])
        a.compute_activations_feature_map_in(train,test)

def gradients_input():
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = params_dict_intel['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = Intel(0.0,'test')
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

    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient_input',params_dict_intel,m_names[i])
        a.compute_gradients_input(train,test)

def feature_map_per_layer():
    
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = 1
    params_dict_intel['max_epochs'] = 5
    print('Loading Data... \n')
    data = Intel(0.0,'test')
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
    
    m = [CNN_INTEL_TANH()]
    m_names = ['tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_intel,m_names[i])
        a.compute_activations_feature_map_per_layer(train,test)

def feature_map_grad():
    
    batch_size_train = params_dict_intel['batch_size']
    batch_size_test = 1
    params_dict_intel['max_epochs'] = 5
    print('Loading Data... \n')
    data = Intel(0.0,'test')
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
    
    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish','tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'feature_map',params_dict_intel,m_names[i])
        a.compute_gradients_feature_map_in(train,test)

def gradients_input_output_all_layers():
    batch_size_train = 1
    batch_size_test = params_dict_intel['batch_size']

    acc = {}
    loss = {}

    print('Loading Data... \n')
    data = Intel(0.0,'test')

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
    m = [CNN_INTEL_RELU(),CNN_INTEL_SWISH(),CNN_INTEL_TANH()]
    m_names = ['relu','swish', 'tanh']

    print('Before test start make sure that you have set the correct parameters')
    input('Press any key to continue...')

    for i, model in enumerate(m):
        print(f'Training CNN with activation function [{m_names[i]}]')
        
        a = ActivationFunction(model,'gradient_input',params_dict_intel,m_names[i])
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
    #feature_map_per_layer()
    #feature_map_grad()
    #gradients_input_output_all_layers()
    
        
if __name__== "__main__":

    main()