
#default dependencies
import numpy as np
import sys
import os

#special dependencies
import torch
import torch.nn as nn

from helper import prepare_device,prepare_summary_writer
from helper import plot_loss_acc,get_layer_names,get_gradients,plot
from helper import get_activations

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

class ActivationFunction:
    def __init__(self,model,param,dict,act_fn,num_classes,**kwargs):
        self.model = model
        self.dict = {
            'act_fn' : act_fn,
            'lr' : dict[f'lr_{act_fn}'],
            'bs' : dict['batch_size'],
            'epochs' : dict['max_epochs'],
            'classes' : num_classes,
            'weight_decay' : dict['weight_decay']
        }
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.dict['lr'],weight_decay=self.dict['weight_decay'])
        self.writer = prepare_summary_writer(param,act_fn)
        self.device = prepare_device()
           
    def __evaluate(self,X, y, train=False):
        if train:
            self.model.zero_grad()
        scores = self.model(X)
        matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,y)]
        acc = matches.count(True)/len(matches)

        loss = self.criterion(scores,y)
        if train:
            loss.backward()
            self.optimizer.step()
        return acc,loss.item()

    def __train(self,epoch,train):
        self.model.train()
        acc = 0.0
        loss = 0.0
        for idx,(data,targets) in enumerate(train):
            
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            accs, losses = self.__evaluate(data,targets,train=True)
            loss += losses * data.size(0)
            acc += accs * data.size(0)
        
        mean_acc = acc / len(train.dataset)
        mean_loss = loss / len(train.dataset)
        self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
        self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)
        return mean_acc,mean_loss
    
    def __test(self,epoch,test):
        acc = 0.0
        loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for (data,targets) in test:
                data = data.to(self.device)
                targets = targets.to(self.device)

                accs, losses = self.__evaluate(data,targets,train=False)
                loss += losses * data.size(0)
                acc += accs * data.size(0)
        mean_acc = acc / len(test.dataset)
        mean_loss = loss / len(test.dataset)
        self.writer.add_scalar('Mean Accuracy Test',mean_acc,epoch)
        print(epoch)
        self.writer.add_scalar('Mean Loss Test',mean_loss,epoch)
        return mean_acc,mean_loss
    
    def compute(self,train,test,print=True):
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        n_epochs = int(self.dict['epochs'])
        self.model.to(self.device)
        for epoch in range(n_epochs):
            acc,loss = self.__train(epoch,train)
            train_acc.append(acc)
            train_loss.append(loss)
            acc,loss = self.__test(epoch,test)
            test_acc.append(acc)
            test_loss.append(loss)
        if print:
            plot_loss_acc(train_acc,test_acc,train_loss,test_loss,n_epochs,self.dict['act_fn'])

    def compute_gradients(self,train,test):
        dict = {}
        dict_class_1 = {}
        dict_class_2 = {}
        dict_class_3 = {}
        dict_class_4 = {}
        dict_class_5 = {}
        dict_class_6 = {}
        dict_class_7 = {}
        dict_class_8 = {}
        dict_class_9 = {}
        dict_class_10 = {}

        name_layers = get_layer_names(self.model)
        for i,name in enumerate(name_layers):
                dict_class_1[name] = []
                dict_class_2[name] = []
                dict_class_3[name] = []
                dict_class_4[name] = []
                dict_class_5[name] = []
                dict_class_6[name] = []
                dict_class_7[name] = []
                dict_class_8[name] = []
                dict_class_9[name] = []
                dict_class_10[name] = []
  
        self.model.to(self.device)
        n_epochs = self.dict['epochs']
        for epoch in range(self.dict['epochs']):
            print(f'Epoch: [{epoch+1} / {n_epochs}]')
            self.model.train()
            for (data,targets) in train:
                data = data.to(self.device)
                targets = targets.to(self.device)
                    
                self.model.zero_grad()
                scores = self.model(data)

                loss = self.criterion(scores,targets)
                loss.backward()
                self.optimizer.step()
                gradient = get_gradients(self.model,'weight')
                t = 0
                for i,idx in enumerate(targets[0]):
                    if idx == 1:
                        t = i
                for key in gradient.keys():
                    grad = gradient[key]
                    if t == 0:
                        dict_class_1[key] = np.append(dict_class_1[key],grad)
                    if t == 1:
                        dict_class_2[key] = np.append(dict_class_2[key],grad)
                    if t == 2:
                        dict_class_3[key] = np.append(dict_class_3[key],grad)
                    if t == 3:
                        dict_class_4[key] = np.append(dict_class_4[key],grad)
                    if t == 4:
                        dict_class_5[key] = np.append(dict_class_5[key],grad)
                    if t == 5:
                        dict_class_6[key] = np.append(dict_class_6[key],grad)
                    if t == 6:
                        dict_class_7[key] = np.append(dict_class_7[key],grad)
                    if t == 7:
                        dict_class_8[key] = np.append(dict_class_8[key],grad)
                    if t == 8:
                        dict_class_9[key] = np.append(dict_class_9[key],grad)
                    if t == 9:
                        dict_class_10[key] = np.append(dict_class_10[key],grad)                                        
            with torch.no_grad():
                acc = 0.0
                loss = 0.0
                for (data,targets) in test:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    accs, losses = self.__evaluate(data,targets,train=False)
                    loss += losses * data.size(0)
                    acc += accs * data.size(0)
            mean_acc = acc / len(test.dataset)
            mean_loss = loss / len(test.dataset)
            self.writer.add_scalar('Mean Accuracy Test',mean_acc,epoch)
            self.writer.add_scalar('Mean Loss Test',mean_loss,epoch)
            
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6,dict_class_7,dict_class_8,dict_class_9,dict_class_10]
            row = len(dict_class_1.keys()) /2
        
            for idx,n in enumerate(l):
                plot_dict = {
                'row' : row,
                'col' : 2,
                'bins': 50,
                'label': f'Gradient distribution for class {idx+1}',
                'type' : 'hist',
                }
                plot(n,plot_dict)
    
    def compute_activations(self,train,test):
        dict = {}
        dict_class_1 = {}
        dict_class_2 = {}
        dict_class_3 = {}
        dict_class_4 = {}
        dict_class_5 = {}
        dict_class_6 = {}
        dict_class_7 = {}
        dict_class_8 = {}
        dict_class_9 = {}
        dict_class_10 = {}

        name_layers = get_layer_names(self.model)
        for i,name in enumerate(name_layers):
                dict_class_1[name] = []
                dict_class_2[name] = []
                dict_class_3[name] = []
                dict_class_4[name] = []
                dict_class_5[name] = []
                dict_class_6[name] = []
                dict_class_7[name] = []
                dict_class_8[name] = []
                dict_class_9[name] = []
                dict_class_10[name] = []
  
        self.model.to(self.device)
        n_epochs = self.dict['epochs']
        for epoch in range(self.dict['epochs']):
            print(f'Epoch: [{epoch+1} / {n_epochs}]')
            self.model.train()
            for (data,targets) in train:
                data = data.to(self.device)
                targets = targets.to(self.device)
                    
                self.model.zero_grad()
                scores = self.model(data)

                loss = self.criterion(scores,targets)
                loss.backward()
                self.optimizer.step()
            for (data,targets) in test:
                data = data.to(self.device)
                targets = targets.to(self.device)

                activation = get_activations(self.model,data)
                t = 0
                for i,idx in enumerate(targets[0]):
                    if idx == 1:
                        t = i
                for key in activation.keys():
                    grad = activation[key]
                    if t == 0:
                        dict_class_1[key] = np.append(dict_class_1[key],grad)
                    if t == 1:
                        dict_class_2[key] = np.append(dict_class_2[key],grad)
                    if t == 2:
                        dict_class_3[key] = np.append(dict_class_3[key],grad)
                    if t == 3:
                        dict_class_4[key] = np.append(dict_class_4[key],grad)
                    if t == 4:
                        dict_class_5[key] = np.append(dict_class_5[key],grad)
                    if t == 5:
                        dict_class_6[key] = np.append(dict_class_6[key],grad)
                    if t == 6:
                        dict_class_7[key] = np.append(dict_class_7[key],grad)
                    if t == 7:
                        dict_class_8[key] = np.append(dict_class_8[key],grad)
                    if t == 8:
                        dict_class_9[key] = np.append(dict_class_9[key],grad)
                    if t == 9:
                        dict_class_10[key] = np.append(dict_class_10[key],grad)
            
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6,dict_class_7,dict_class_8,dict_class_9,dict_class_10]
            row = len(dict_class_1.keys()) /2
            
            for idx,n in enumerate(l):
                plot_dict = {
                'row' : row,
                'col' : 2,
                'bins': 50,
                'label': f'Activation values {idx+1} Epoch: {epoch+1}',
                'type' : 'hist',
                }
                plot(n,plot_dict)
    
from datasets.data import MNIST
from datasets.data import CIFAR10
from datasets.data import Intel
from datasets.datasets import CustomDataset

from models.cifar import CNN_CIFAR_RELU,CNN_CIFAR_SWISH,CNN_CIFAR_TANH

from parameters import params_dict_cifar
def main():

    #define param here
    batch_size = params_dict_cifar['batch_size']

    #load data
    print('Loading Data... \n')
    data = CIFAR10(0.1,'validate')
    data.prepare_data()
    m,s = data.get_mean_std()
    print('Done.')
    
    #load into dataloader
    dataset = CustomDataset

    print('Loading train and test samples into DataLoader... \n')
    X_train,y_train = data.get_data('train')
    X_test, y_test = data.get_data('val')
    train = dataset(X_train,y_train)
    test = dataset(X_test,y_test)
    train.set_mean_std(m,s)
    test.set_mean_std(m,s)
    train = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    test = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
    m = [CNN_CIFAR_RELU(),CNN_CIFAR_SWISH(),CNN_CIFAR_TANH()]
    act = ['relu','swish','tanh']
    for i,model in enumerate(m):
        a = ActivationFunction(model,'',params_dict_cifar,act[i],10)
        a.compute(train,test,True)
    #a.compute_gradients(train,test)
    
        
if __name__== "__main__":

    main()