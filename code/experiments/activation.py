
#default dependencies
import numpy as np
import sys
import os

#special dependencies
import torch
import torch.nn as nn

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from experiments.helper import prepare_device,prepare_summary_writer
from experiments.helper import plot_single,get_layer_names,get_gradients,plot
from experiments.helper import get_activations, set_hook,activations


class ActivationFunction:
    def __init__(self,model,param,dict,act_fn,num_classes,extra):
        self.model = model
        self.dict = {
            'act_fn' : act_fn,
            'lr' : dict[f'lr_{act_fn}'],
            'epochs' : dict['max_epochs'],
            'classes' : dict['classes'],
            'weight_decay' : dict['weight_decay']
        }
        #Initialisierung Loss und Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.dict['lr'],weight_decay=self.dict['weight_decay'])
        
        #Initialisierung writer und GPU
        self.writer = prepare_summary_writer(param,act_fn,extra)
        self.device = prepare_device()
        print('Loaded ActivationFunction class with parameters:')
        print('Key:\n')
        for k in self.dict.keys():
            print(f'\t{k}: \t{self.dict[k]}')
           
    #Hier wird Accuracy und Loss berechnet abhängig ob training oder testzyklus
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

    #train funktion die den Writer aufruft und mittelwert von Accuracy und Loss des Batches berechnet
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
    
    #test funktion die den Writer aufruft und mittelwert von Accuracy und Loss des Batches berechnet
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
        self.writer.add_scalar('Mean Loss Test',mean_loss,epoch)
        return mean_acc,mean_loss
    
    #Funktion die Training, Test uebernimmt
    def compute(self,train,test,patience):
        best_loss = 100
        trigger = 0 
        stop = False #benoetigt fuer early stopping
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
            curr_loss = loss

            if (best_loss > curr_loss): #Kontrolle ob aktueller loss schlechter als vorheriger
              best_loss = curr_loss
            
            if (curr_loss > best_loss): #patience
              trigger +=1
              print(f'curr_loss: {curr_loss}, best_loss: {best_loss}')
              if (trigger >= patience): #wenn patience überschritten wird abbruch training
                print(f'Early Stopping! Measuring last val Accuracy and loss')
                stop = True
            else:
              trigger = 0
            if stop == True:
              break
        self.dict['epochs'] = len(train_acc)

        plot_single(train_acc,test_acc,train_loss,test_loss,self.dict)
        

    def compute_gradients_per_class(self,train,test):
        best_loss = 100
        trigger = 0 
        stop = False #benoetigt fuer early stopping
        dict = {}
        dict_class_1 = {}
        dict_class_2 = {}
        dict_class_3 = {}
        dict_class_4 = {}
        dict_class_5 = {}
        dict_class_6 = {}
        if self.dict['classes'] > 6:
            dict_class_7 = {}
            dict_class_8 = {}
            dict_class_9 = {}
            dict_class_10 = {}

        name_layers = get_layer_names(self.model)
        for i,name in enumerate(name_layers):
            if 'bn' not in name:
                if 'pool'  not in name : 
                    if 'drop' not in name :
                        dict_class_1[name] = []
                        dict_class_2[name] = []
                        dict_class_3[name] = []
                        dict_class_4[name] = []
                        dict_class_5[name] = []
                        dict_class_6[name] = []
                        if self.dict['classes'] > 6:
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
                acc = 0.0
                loss = 0.0
                
                data = data.to(self.device)
                targets = targets.to(self.device)
                    
                self.model.zero_grad()
                scores = self.model(data)
                
                matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,targets)]
                accs = matches.count(True)/len(matches)
                losses = self.criterion(scores,targets)
                losses.backward()
                loss += losses * data.size(0)
                acc += accs * data.size(0)
                
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
                    
                    if self.dict['classes'] > 6:
                        if t == 6:
                            dict_class_7[key] = np.append(dict_class_7[key],grad)
                        if t == 7:
                            dict_class_8[key] = np.append(dict_class_8[key],grad)
                        if t == 8:
                            dict_class_9[key] = np.append(dict_class_9[key],grad)
                        if t == 9:
                            dict_class_10[key] = np.append(dict_class_10[key],grad)                                        
            mean_acc = acc / 1
            mean_loss = loss /1
            self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
            self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)
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
            #dict_class_7,dict_class_8,dict_class_9,dict_class_10
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6]
            row = (len(dict_class_1.keys()) /2) +1

            for idx,n in enumerate(l):
                plot_dict = {
                'row' : int(row),
                'col' : 2,
                'bins': 100,
                'label': f'Gradient distribution for class {idx+1}',
                'type' : 'hist',
                'act_fn': self.dict['act_fn'],
                'lr' : self.dict['lr'],
                'class' : {idx+1}
                }
                plot(n,plot_dict,epoch)
            
                for key in n.keys():
                    n[key] = []
            
            curr_loss = mean_loss
            if (best_loss > curr_loss): #Kontrolle ob aktueller loss schlechter als vorheriger
              best_loss = curr_loss
            
            if (curr_loss > best_loss): #patience
              trigger +=1
              print(f'curr_loss: {curr_loss}, best_loss: {best_loss}')
              if (trigger >= 4): #wenn patience überschritten wird abbruch training
                print(f'Early Stopping! Measuring last val Accuracy and loss')
                stop = True
            else:
              trigger = 0
            if stop == True:
              break

    def compute_gradients(self,train,test):
        best_loss = 100
        trigger = 0 
        stop = False #benoetigt fuer early stopping
        dict_class = {}
        name_layers = get_layer_names(self.model)
        for i,name in enumerate(name_layers):
            if 'bn' not in name:
                if 'pool'  not in name : 
                    if 'drop' not in name :
                        dict_class[name] = []
  
        self.model.to(self.device)
        n_epochs = self.dict['epochs']
        
        for epoch in range(self.dict['epochs']):
            for k in dict_class.keys():
              dict_class[k] = []
            
            print(f'Epoch: [{epoch+1} / {n_epochs}]')
            self.model.train()
            acc = 0.0
            loss = 0.0
            for (data,targets) in train:
                
                
                data = data.to(self.device)
                targets = targets.to(self.device)
         
                accs, losses = self.__evaluate(data,targets,train=True)
                loss += losses * data.size(0)
                acc += accs * data.size(0)
                
            
                gradient = get_gradients(self.model,'weight')
                
                for k in gradient.keys():
                  dict_class[k] = np.append(dict_class[k],gradient[k])
                                                 
            mean_acc = acc / len(train.dataset)
            mean_loss = loss / len(train.dataset)
            self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
            self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)

            acc,loss = self.__test(epoch,test)
            
            l = [dict_class]
            row = (len(dict_class.keys()) /2) +1

            for idx,n in enumerate(l):
                plot_dict = {
                'row' : int(row),
                'col' : 2,
                'bins': 100,
                'label': f'Gradient distribution for class {idx+1}',
                'type' : 'hist',
                'act_fn': self.dict['act_fn'],
                'lr' : self.dict['lr'],
                'class' : 'all'
                }
                plot(n,plot_dict,epoch)
            
                for key in n.keys():
                    n[key] = []
            
            curr_loss = loss
            if (best_loss > curr_loss): #Kontrolle ob aktueller loss schlechter als vorheriger
              best_loss = curr_loss
            
            if (curr_loss > best_loss): #patience
              trigger +=1
              print(f'curr_loss: {curr_loss}, best_loss: {best_loss}')
              if (trigger >= 10): #wenn patience überschritten wird abbruch training
                print(f'Early Stopping! Measuring last val Accuracy and loss')
                stop = True
            else:
              trigger = 0
            if stop == True:
              break
            
    def compute_activations(self,train,test):
        best_loss = 100
        trigger = 0 
        stop = False #benoetigt fuer early stopping
        global activations
        dict = {}
        dict_class = {}
        activations.clear()
        name_layers = get_layer_names(self.model)
        for i,name in enumerate(name_layers):
                dict_class[name] = []

        self.model.to(self.device)
        n_epochs = self.dict['epochs']
        for epoch in range(self.dict['epochs']):
            activations.clear()
            accs = 0.0
            loss = 0.0
            print(f'Epoch: [{epoch+1} / {n_epochs}]')
            self.model.train()
            
            for k in dict_class.keys():
              dict_class[k] = []
            
            a,b = self.__train(epoch,train)
            
            set_hook(self.model)
            acc, loss = self.__test(epoch,test) 
            print('before append')        
            for key in activations.keys():
                grad = activations[key]
                if key != '':
                  dict_class[key] = np.append(dict_class[key],grad)
            print('after append')
            l = [dict_class]
            row = (len(dict_class.keys()) /2)

            for idx,n in enumerate(l):
                plot_dict = {
                'row' : int(row),
                'col' : 2,
                'bins': 100,
                'label': f'Activation values for class {idx+1}',
                'type' : 'acti',
                'act_fn': self.dict['act_fn'],
                'lr' : self.dict['lr'],
                'class' : 'all'
                }
                plot(n,plot_dict,epoch)
            
                for key in n.keys():
                    n[key] = []

            curr_loss = loss
            if (best_loss > curr_loss): #Kontrolle ob aktueller loss schlechter als vorheriger
              best_loss = curr_loss
            
            if (curr_loss > best_loss): #patience
              trigger +=1
              print(f'curr_loss: {curr_loss}, best_loss: {best_loss}')
              if (trigger >= 10): #wenn patience überschritten wird abbruch training
                print(f'Early Stopping! Measuring last val Accuracy and loss')
                stop = True
            else:
              trigger = 0
            if stop == True:
              break