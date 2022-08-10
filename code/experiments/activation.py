
#default dependencies
import numpy as np
import sys
import os
import logging
from sklearn.ensemble import GradientBoostingClassifier
#special dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from experiments.helper import prepare_device,prepare_summary_writer
from experiments.helper import plot_single,get_layer_names,get_gradients,plot
from experiments.helper import get_activations, set_hook, set_hook_feature_map,activations,gradients
from experiments.helper import set_hook_feature_map_in,set_hook_in
from experiments.helper import set_backward_hook,set_backward_hook_out
from experiments.helper import gradients_in
from experiments.helper import set_backward_hook_in_out_all_layers,set_backward_hook_out_all_layers
from experiments.helper import set_backward_hook_in_out
#from experiments.parameters import params_dict_mnist

class ActivationFunction:
  def __init__(self,model,param,dict,dict_algo,act_fn,method,num):
    self.num = num
    self.model = model #aktuell zu trainierende modell
    self.method = method
    self.act_fn = act_fn
    self.lv = 1
    self.dict = dict
    self.dict_algo = dict_algo
    lr = self.dict_algo[f'lr_{act_fn}_{method}']
    print(lr)
    #Initialisierung Loss und Optimizer
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=self.dict['weight_decay'])
    
    #Initialisierung writer und GPU
    self.writer = prepare_summary_writer(param,act_fn)
    self.device = prepare_device()
    print('Loaded ActivationFunction class with parameters:')
    print('Key:\n')
    for k in self.dict.keys():
        print(f'\t{k}: \t{self.dict[k]}')
           
  #Hier wird Accuracy und Loss berechnet abhängig ob training oder testzyklus
  def __evaluate(self,X, y, train=False):
    if train:
      self.model.zero_grad()  #reset der gradienten während trainingslauf
    scores = self.model(X)  #predicted labels des modells
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,y)] #vergleich predicted mit tatsächlichen werten
    #matches ist array mit True wenn label correct oder False wenn label Falsch
    acc = matches.count(True)/len(matches) #berechnen Accuracy

    loss = self.criterion(scores,y) #berechnen loss
    if train:
      loss.backward()
      self.optimizer.step()
    return acc,loss.item()

  #train funktion die den Writer aufruft und mittelwert von Accuracy und Loss des Batches berechnet
  def __train(self,epoch,train):
      self.model.train()
      acc = 0.0 
      loss = 0.0
      for (data,targets) in train:
          
        data = data.to(self.device)
        targets = targets.to(self.device)
        
        accs, losses = self.__evaluate(data,targets,train=True)
        loss += losses * data.size(0) #aufaddieren loss und acc pro batch
        acc += accs * data.size(0)
      
      mean_acc = acc / len(train.dataset) #berchnung mittlere accuracy des batches
      mean_loss = loss / len(train.dataset)
      self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
      self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)
      return mean_acc,mean_loss
    
  def _prob(self,x,mode):
    dict = self.dict_algo
    if mode == 'drop_cur':
      act_fn = self.act_fn
      gamma = dict[f'cur_{act_fn}']
      if -0.5 * np.exp(-gamma*x) + 1 < 0.9:
        return -0.5 * np.exp(-gamma*x) + 1
      else:
        return 0.9
    if mode == 'drop_ann':
      act_fn = self.act_fn
      gamma = dict[f'ann_{act_fn}']
      d = 0.5      
      if (1.-d)* np.exp(- gamma * x) < 0.1:
        return 0.1
      else:
        return (1.-d)* np.exp(- gamma * x)
    if mode == 'drop_log':
      act_fn = self.act_fn
      gamma = dict[f'log_{act_fn}']
      d = 0.25
      if 0.5 / np.exp(gamma)*(np.log(x))+d < 0.9:
        return 0.5 / np.exp(gamma)*(np.log(x))+d
      else:
        return 0.9
    if mode == 'normal':
      d = 0.8
      return d

  def __train_batch(self,epoch,train,mode):
    self.model.train()
    acc = 0.0 
    loss = 0.0
    for (data,targets) in train:
        
      drop = self.model.update(self._prob(self.lv,mode))
      self.writer.add_scalar('Dropout rate',drop,self.lv)

      self.lv = self.lv + 1
      data = data.to(self.device)
      targets = targets.to(self.device)
      
      accs, losses = self.__evaluate(data,targets,train=True)
      loss += losses * data.size(0) #aufaddieren loss und acc pro batch
      acc += accs * data.size(0)
    
    mean_acc = acc / len(train.dataset) #berchnung mittlere accuracy des batches
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
    
  def __train_grad(self,epoch,train):
    self.model.train()
    acc = 0.0 
    loss = 0.0
    act_fn = self.dict['act_fn']
    gradients_ = {}
    for idx,(data,targets) in enumerate(train):
        
      data = data.to(self.device)
      targets = targets.to(self.device)
      
      accs, losses = self.__evaluate(data,targets,train=True)
      loss += losses * data.size(0) #aufaddieren loss und acc pro batch
      acc += accs * data.size(0)
    
      if idx % 100 == 0 and idx > 0:
        for key in gradients.keys():
          g = gradients[key]
          gradients_[key] = g
        row = len(gradients_.keys())
        plot_dict = {
          'row' : int(row), #int da fehler bei float
          'col' : int(1),
          'bins': 300,
          'label': f'Gradient distribution input for {act_fn}',
          'type' : 'hist',
          'act_fn': self.dict['act_fn'],
          'lr' : self.dict['lr'],
          'class' : f'all_{idx}_out'
          }
        plot(gradients_,plot_dict,epoch)                             
        


        for key in gradients_in.keys():
          g = gradients_in[key]
          gradients_[key] = g
        row = len(gradients_.keys())
        plot_dict = {
          'row' : int(row), #int da fehler bei float
          'col' : int(1),
          'bins': 300,
          'label': f'Gradient distribution input for {act_fn}',
          'type' : 'hist',
          'act_fn': self.dict['act_fn'],
          'lr' : self.dict['lr'],
          'class' : f'all_{idx}_in'
          }
        plot(gradients_,plot_dict,epoch)                             
        gradients_.clear()
    mean_acc = acc / len(train.dataset) #berchnung mittlere accuracy des batches
    mean_loss = loss / len(train.dataset)
    self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
    self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)
    return mean_acc,mean_loss

    #Funktion die Training, Test uebernimmt
  def compute(self,train,test,patience,plot=False):
    best_loss = 100 #benötigt für early stop
    trigger = 0 #benötigt für early stop
    stop = False #benötigt für early stop
    
    train_acc = [] #benötigt für plot von acc und loss
    test_acc = [] #benötigt für plot von acc und loss
    train_loss = [] #benötigt für plot von acc und loss
    test_loss = [] #benötigt für plot von acc und loss
    
    n_epochs = int(self.dict['epochs']) #fehler wenn self.dict[epochs] in range funktion war
    self.model.to(self.device)
    
    for epoch in range(n_epochs):
        acc,loss = self.__train(epoch,train)
        train_acc.append(acc)
        train_loss.append(loss)
        
        acc,loss = self.__test(epoch,test)
        test_acc.append(acc)
        test_loss.append(loss)
        curr_loss = loss #Speicherung aktuellen loss für early stop

        if (best_loss > curr_loss): #Kontrolle ob aktueller loss schlechter als vorheriger
          best_loss = curr_loss
        
        if (curr_loss > best_loss): #patience
          trigger +=1
          print(f'curr_loss: {curr_loss}, best_loss: {best_loss}')
          if (trigger >= patience): #wenn patience überschritten wird abbruch training
            print(f'Early Stopping! Measuring last val Accuracy and loss')
            stop = True
        else:
          trigger = 0 #reset patience, da neues optimum
        if stop == True:
          break #Abbruch da patience überschritten
    self.dict['epochs'] = len(train_acc)  #wegen early stop muss epoch angepasst werden sonst fehler bei plot

    if plot:
      plot_single(train_acc,test_acc,train_loss,test_loss,self.dict)
        
  def compute_drop_sched(self,train,test,patience):
      best_loss = 100
      trigger = 0 
      stop = False #benoetigt fuer early stopping
      train_acc = []
      test_acc = []
      train_loss = []
      test_loss = []
      
      n_epochs = int(self.dict['epochs'])
      self.model.to(self.device)
      i = 0.0
      for epoch in range(n_epochs):
          if epoch % 1 == 0:
            drop = self.model.update_drop(epoch)
            print(f'Dropout rate = {drop}')
    
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
    
  def compute_drop_sched_batch(self,train,test,patience,mode,log=False):
      best_loss = 100
      trigger = 0 
      stop = False #benoetigt fuer early stopping
      train_acc = []
      test_acc = []
      train_loss = []
      test_loss = []
      
      n_epochs = int(self.dict['max_epochs'])
      self.model.to(self.device)
      i = 0.0
      for epoch in range(n_epochs):
          
          acc,loss = self.__train_batch(epoch,train,mode)
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
      if log == True:
        model = self.dict['model']
        act_fn = self.act_fn
        f = open(f'test_acc_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        ff = open(f'train_acc_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        fff = open(f'test_loss_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        ffff = open(f'train_loss_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        for lv,i in enumerate(test_acc):
          f.write(f'{i}\n')
          ff.write(f'{train_acc[lv]}\n')
          fff.write(f'{test_loss[lv]}\n')
          ffff.write(f'{train_loss[lv]}\n')
        f.close()
        ff.close()
        fff.close()
        ffff.close()
        self.lv += 1

  def compute_drop_sched_batch_early_stop(self,train,test,patience,mode,log=False):
      best_loss = 100
      trigger = 0 
      stop = False #benoetigt fuer early stopping
      train_acc = []
      test_acc = []
      train_loss = []
      test_loss = []
      
      n_epochs = int(self.dict['max_epochs'])
      self.model.to(self.device)
      i = 0.0
      for epoch in range(n_epochs):
          
          acc,loss = self.__train_batch(epoch,train,mode)
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
            if trigger == 1:
              model = self.dict['model']        
              torch.save(model.state_dict(), f'saved_models/{self.method}_{model}_{self.act_fn}_{self.num}')
            print(f'curr_loss: {curr_loss}, best_loss: {best_loss}')
            if (trigger >= patience): #wenn patience überschritten wird abbruch training
              print(f'Early Stopping! Measuring last val Accuracy and loss')
              stop = True
          else:
            trigger = 0
          if stop == True:
            break
      if log == True:
        model = self.dict['model']
        act_fn = self.act_fn
        f = open(f'test_acc_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        ff = open(f'train_acc_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        fff = open(f'test_loss_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        ffff = open(f'train_loss_{act_fn}_{model}_{self.method}_{self.num}.txt','w+')
        for lv,i in enumerate(test_acc):
          f.write(f'{i}\n')
          ff.write(f'{train_acc[lv]}\n')
          fff.write(f'{test_loss[lv]}\n')
          ffff.write(f'{train_loss[lv]}\n')
        f.close()
        ff.close()
        fff.close()
        ffff.close()
        self.lv += 1      
    
  def compute_gradients_per_class(self,train,test,patience):
      best_loss = 100
      trigger = 0 
      stop = False #benoetigt fuer early stopping
      dict_class_1 = {}
      dict_class_2 = {}
      dict_class_3 = {}
      dict_class_4 = {}
      dict_class_5 = {}
      dict_class_6 = {}
      if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
          dict_class_7 = {}
          dict_class_8 = {}
          dict_class_9 = {}
          dict_class_10 = {}

      name_layers = get_layer_names(self.model) #Liste aller Namen der einzelnen Layer und Aktivierungsfunktion
      for i,name in enumerate(name_layers):
          if 'bn' not in name:  #gab fehler bei not in mit or anweisung; aufteilung 
              if 'pool'  not in name : 
                  if 'drop' not in name :
                      dict_class_1[name] = [] #Batchnorm pool und enthalten keine partiellen abl.
                      dict_class_2[name] = [] #drop uninteressant
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
              #Nach berechnung können aktuelle Gradienten der einzelnen Layer rausgezogen werden
              
              t = 0 #Benötigt für Klassenaufteilung
              for i,idx in enumerate(targets[0]):
                  if idx == 1:
                      t = i #Ermittlung aktuellen Labels des Trainingsdatums 
              for key in gradient.keys(): #Zuweisung des Gradienten zur korrekten Klasse
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
          mean_acc = acc / len(train.dataset)
          mean_loss = loss /len(train.dataset)
          self.writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
          self.writer.add_scalar('Mean Loss Train',mean_loss,epoch)
          
          acc,loss = self.__test(epoch,test)
          l = []
          
          if self.dict['classes'] > 6: #Intel nur 6 Klassen deshalb differenzierung nötig
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6,dict_class_7,dict_class_8,dict_class_9,dict_class_10]
          else:
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6]
          
          row = (len(dict_class_1.keys()) /2) +1  #Berechung wieviele Reihen für Plot

          for idx,n in enumerate(l):
              plot_dict = {
              'row' : int(row), #int da fehler bei float
              'col' : 2,
              'bins': 120,
              'label': f'Gradient distribution for class {idx+1}',
              'type' : 'hist',
              'act_fn': self.dict['act_fn'],
              'lr' : self.dict['lr'],
              'class' : {idx+1}
              }
              plot(n,plot_dict,epoch)
          
          curr_loss = mean_loss
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

  def compute_gradients(self,train,test,patience): #Berechnung Gradient ohne Klassenaufteilung
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
              
          
              #gradient = get_gradients(self.model,'weight')
              
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
          
  def compute_activations(self,train,test):
      best_loss = 100
      trigger = 0 
      stop = False #benoetigt fuer early stopping
      global activations  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      dict_class = {}
      activations.clear() #reset des Dictionary
      name_layers = get_layer_names(self.model)
      for i,name in enumerate(name_layers):
              dict_class[name] = []

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      for epoch in range(self.dict['epochs']):
          activations.clear() #reset des Dictionary pro Epoche
          acc = 0.0
          loss = 0.0
          print(f'Epoch: [{epoch+1} / {n_epochs}]')
          self.model.train()
          
          for k in dict_class.keys():
            dict_class[k] = []
          
          _,_ = self.__train(epoch,train)
          
          set_hook(self.model)  #hook setzen um aktivierungswerte rauszuziehen
          acc, loss = self.__test(epoch,test) 

          for key in activations.keys():
              grad = activations[key]
              if key != '': #gab fehler weil ein eintrag zuviel in dictionary war
                dict_class[key] = np.append(dict_class[key],grad)

          l = [dict_class] #Liste für for loop
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
  
  def compute_activations_feature_map(self,train,test):
      global activations  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      dict_class = {}
      activations.clear() #reset des Dictionary
      name_layers = get_layer_names(self.model)
      for name in name_layers:
          dict_class[name] = []

      self.model.to(self.device)
      
      n_epochs = self.dict['epochs']
      for epoch in range(self.dict['epochs']):
          activations.clear() #reset des Dictionary pro Epoche
          acc = 0.0
          loss = 0.0
          print(f'Epoch: [{epoch+1} / {n_epochs}]')
          self.model.train()
          
          for k in dict_class.keys():
            dict_class[k] = []
          
          _,_ = self.__train(epoch,train)
          
          set_hook_feature_map(self.model)  #hook setzen um aktivierungswerte rauszuziehen
          
          data,target = next(iter(test))

          self.model.eval()
          with torch.no_grad():
              data = data.to(self.device)
              target = target.to(self.device)
              scores = self.model(data)  #predicted labels des modells
              matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,target)] #vergleich predicted mit tatsächlichen werten
              #matches ist array mit True wenn label correct oder False wenn label Falsch
              acc = matches.count(True)/len(matches) #berechnen Accuracy
              loss = self.criterion(scores,target) #berechnen loss

          act = []
          processed = []
          names = []
          for key in activations.keys():
            if 'conv' in key:

              act = activations[key].squeeze()

              tf = torch.from_numpy(act)

              im = torch.sum(tf,0)
              im = im / tf.shape[0]

              processed.append(im.data.cpu().numpy())
              names.append(key)
    
          fig = plt.figure(figsize=(30, 50))
          for i in range(len(processed)):
                  a = fig.add_subplot(5, 4, i+1)
                  imgplot = plt.imshow(processed[i])
                  a.axis("off")
                  a.set_title(names[i])
          act_fn = self.dict['act_fn']        
          plt.savefig(f'intel_{epoch}_feature_map_{act_fn}.pdf')
          plt.clf()
          plt.cla()
          plt.close()
  
  def compute_activations_feature_map_per_layer(self,train,test):
      global activations  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      dict_class = {}
      activations.clear() #reset des Dictionary
      name_layers = get_layer_names(self.model)
      for name in name_layers:
          dict_class[name] = []

      self.model.to(self.device)
      
      n_epochs = self.dict['epochs']
      for epoch in range(self.dict['epochs']):
          activations.clear() #reset des Dictionary pro Epoche
          acc = 0.0
          loss = 0.0
          print(f'Epoch: [{epoch+1} / {n_epochs}]')
          self.model.train()
          
          for k in dict_class.keys():
            dict_class[k] = []
          
          _,_ = self.__train(epoch,train)
          
          set_hook_feature_map(self.model)  #hook setzen um aktivierungswerte rauszuziehen
          
          data,target = next(iter(test))

          self.model.eval()
          with torch.no_grad():
              data = data.to(self.device)
              target = target.to(self.device)
              scores = self.model(data)  #predicted labels des modells
              matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,target)] #vergleich predicted mit tatsächlichen werten
              #matches ist array mit True wenn label correct oder False wenn label Falsch
              acc = matches.count(True)/len(matches) #berechnen Accuracy
              loss = self.criterion(scores,target) #berechnen loss

          act = []
          tf = []
          processed = []
          names = []
          for key in activations.keys():
            if 'conv' in key:
              
              act = activations[key].squeeze()
              tf.append(torch.from_numpy(act))
              names.append(key)
          for j,data in enumerate(tf):

            for i in range(data.shape[0]):
              if i < 10:
                fig = plt.figure(figsize=(30, 50))
                a = fig.add_subplot()
                imgplot = plt.imshow(data[i])
                a.axis("off")
                a.set_title(i)
                act_fn = self.dict['act_fn']        
                plt.savefig(f'cifar_{epoch}_feature_map_{act_fn}_{names[j]}_{i}.pdf')
                plt.clf()
                plt.cla()
                plt.close()

  def compute_activations_in(self,train,test):
      best_loss = 100
      trigger = 0 
      stop = False #benoetigt fuer early stopping
      global activations  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      dict_class = {}
      activations.clear() #reset des Dictionary
      name_layers = get_layer_names(self.model)
      for i,name in enumerate(name_layers):
              dict_class[name] = []

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      for epoch in range(self.dict['epochs']):
          activations.clear() #reset des Dictionary pro Epoche
          acc = 0.0
          loss = 0.0
          print(f'Epoch: [{epoch+1} / {n_epochs}]')
          self.model.train()
          
          for k in dict_class.keys():
            dict_class[k] = []
          
          _,_ = self.__train(epoch,train)
          
          set_hook_in(self.model)  #hook setzen um aktivierungswerte rauszuziehen
          acc, loss = self.__test(epoch,test) 

          for key in activations.keys():
              grad = activations[key]
              if key != '': #gab fehler weil ein eintrag zuviel in dictionary war
                dict_class[key] = np.append(dict_class[key],grad)

          l = [dict_class] #Liste für for loop
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
  
  def compute_activations_feature_map_in(self,train,test):
      global activations  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      dict_class = {}
      activations.clear() #reset des Dictionary
      name_layers = get_layer_names(self.model)
      for name in name_layers:
          dict_class[name] = []

      self.model.to(self.device)
      
      n_epochs = self.dict['epochs']
      for epoch in range(self.dict['epochs']):
          activations.clear() #reset des Dictionary pro Epoche
          acc = 0.0
          loss = 0.0
          print(f'Epoch: [{epoch+1} / {n_epochs}]')
          self.model.train()
          
          for k in dict_class.keys():
            dict_class[k] = []
          
          _,_ = self.__train(epoch,train)
  
          set_hook_feature_map_in(self.model)  #hook setzen um aktivierungswerte rauszuziehen
          
          data,target = next(iter(test))
          #data = data.unsqueeze() mnist
          self.model.eval()
          with torch.no_grad():
              data = data.to(self.device)
              target = target.to(self.device)
              scores = self.model(data)  #predicted labels des modells
              matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,target)] #vergleich predicted mit tatsächlichen werten
              #matches ist array mit True wenn label correct oder False wenn label Falsch
              acc = matches.count(True)/len(matches) #berechnen Accuracy
              loss = self.criterion(scores,target) #berechnen loss

          act = []
          processed = []
          names = []
          for key in activations.keys():
            if 'conv' in key:
              act = activations[key].squeeze()
              tf = torch.from_numpy(act)

              im = torch.sum(tf,0)
              im = im / tf.shape[0]

              processed.append(im)
              names.append(key)
    
          
          fig = plt.figure(figsize=(30, 50))
          for i in range(len(processed)):
                  a = fig.add_subplot(5, 4, i+1)
                  imgplot = plt.imshow(processed[i])
                  a.axis("off")
                  a.set_title(names[i])
          act_fn = self.dict['act_fn']        
          plt.savefig(f'intel_{epoch}_feature_map_{act_fn}_in.pdf')
          plt.clf()
          plt.cla()
          plt.close()
  
  def compute_gradients_input(self,train,test):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      gradients.clear() #reset des Dictionary

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      set_backward_hook(self.model)
      for epoch in range(self.dict['epochs']):
        gradients.clear()  
        acc = 0.0
        loss = 0.0
        print(f'Epoch: [{epoch+1} / {n_epochs}]')
        acc,loss = self.__train_grad(epoch,train)  
  
        with torch.no_grad():
            accs, loss = self.__test(epoch,test)
  
  def compute_gradients_feature_map_in(self,train,test):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      dict_class = {}
      gradients.clear() #reset des Dictionary
      name_layers = get_layer_names(self.model)
      for name in name_layers:
          dict_class[name] = []

      self.model.to(self.device)
      
      n_epochs = self.dict['epochs']
      for epoch in range(self.dict['epochs']):
          gradients.clear() #reset des Dictionary pro Epoche
          acc = 0.0
          loss = 0.0
          print(f'Epoch: [{epoch+1} / {n_epochs}]')
          self.model.train()
          
          for k in dict_class.keys():
            dict_class[k] = []
          
          set_backward_hook(self.model)
          _,_ = self.__train(epoch,train)
              
          _,_ = self.__test(epoch,test)
          #data = data.unsqueeze() mnist
          
          act = []
          processed = []
          names = []
          for key in gradients.keys():
            if 'conv' in key:
              act = gradients[key]
              im = torch.sum(act,[0,1])
              print(np.shape(im))
              im = im / act.shape[0]
              processed.append(im.cpu())
              names.append(key)
    
          
          fig = plt.figure(figsize=(30, 50))
          for i in range(len(processed)):
                  a = fig.add_subplot(5, 4, i+1)
                  imgplot = plt.imshow(processed[i])
                  a.axis("off")
                  a.set_title(names[i])
          act_fn = self.dict['act_fn']        
          plt.savefig(f'mnist_{epoch}_feature_map_{act_fn}_grad_in.pdf')
          plt.clf()
          plt.cla()
          plt.close()
  
  def get_gradients_hook(self,train,test):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      gradients.clear() #reset des Dictionary

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      set_backward_hook_out(self.model)
      for epoch in range(self.dict['epochs']):
        gradients.clear()  
        acc = 0.0
        loss = 0.0
        print(f'Epoch: [{epoch+1} / {n_epochs}]')
        acc,loss = self.__train_grad(epoch,train)  
  
        with torch.no_grad():
            accs, loss = self.__test(epoch,test)
  
  
  def compute_gradients_per_class_hook(self,train,test,patience,akt):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      gradients.clear() #reset des Dictionary
      dict_class_1 = {}
      dict_class_2 = {}
      dict_class_3 = {}
      dict_class_4 = {}
      dict_class_5 = {}
      dict_class_6 = {}
      if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
          dict_class_7 = {}
          dict_class_8 = {}
          dict_class_9 = {}
          dict_class_10 = {}

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      set_backward_hook(self.model)
      
      if akt == 0:
        k = ['ReLU()_6', 'ReLU()_5', 'ReLU()_4', 'ReLU()_3', 'ReLU()_2', 'ReLU()_1', 'ReLU()_0']
      if akt == 1:
        k = ['SiLU()_6', 'SiLU()_5', 'SiLU()_4', 'SiLU()_3', 'SiLU()_2', 'SiLU()_1', 'SiLU()_0']
      if akt == 2:
        k = ['Tanh()_6', 'Tanh()_5', 'Tanh()_4', 'Tanh()_3', 'Tanh()_2', 'Tanh()_1', 'Tanh()_0']
      for key in k:
          dict_class_1[key] = []
          dict_class_2[key] = []
          dict_class_3[key] = []
          dict_class_4[key] = []
          dict_class_5[key] = []
          dict_class_6[key] = []
          if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
              dict_class_7[key] = []
              dict_class_8[key] = []
              dict_class_9[key] = []
              dict_class_10[key] = []       
      
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

              t = 0 #Benötigt für Klassenaufteilung
              for i,idx in enumerate(targets[0]):
                  if idx == 1:
                      t = i #Ermittlung aktuellen Labels des Trainingsdatums 
              
              for key in gradients.keys():
                if t == 0:
                    dict_class_1[key].append(gradients[key])
                if t == 1:  
                    dict_class_2[key].append(gradients[key])
                if t == 2:
                    dict_class_3[key].append(gradients[key])
                if t == 3:
                    dict_class_4[key].append(gradients[key])
                if t == 4:
                    dict_class_5[key].append(gradients[key])
                if t == 5:
                    dict_class_6[key].append(gradients[key])
                  
                if self.dict['classes'] > 6:
                    if t == 6:
                        dict_class_7[key].append(gradients[key])
                    if t == 7:
                        dict_class_8[key].append(gradients[key])
                    if t == 8:
                        dict_class_9[key].append(gradients[key])
                    if t == 9:
                        dict_class_10[key].append(gradients[key]) 
          
          for key in dict_class_1.keys():
            temp_list = dict_class_1[key]
            dict_class_1[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_1[key][0])
            dict_class_1[key] = np.sum(dict_class_1[key],axis=0)
            dict_class_1[key] = dict_class_1[key] / anzahl

            temp_list = dict_class_2[key]
            dict_class_2[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_2[key][0])
            dict_class_2[key] = np.sum(dict_class_2[key],axis=0)
            dict_class_2[key] = dict_class_2[key] / anzahl

            temp_list = dict_class_3[key]
            dict_class_3[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_3[key][0])
            dict_class_3[key] = np.sum(dict_class_3[key],axis=0)
            dict_class_3[key] = dict_class_3[key] / anzahl

            temp_list = dict_class_4[key]
            dict_class_4[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_4[key][0])
            dict_class_4[key] = np.sum(dict_class_4[key],axis=0)
            dict_class_4[key] = dict_class_4[key] / anzahl

            temp_list = dict_class_5[key]
            dict_class_5[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_5[key][0])
            dict_class_5[key] = np.sum(dict_class_5[key],axis=0)
            dict_class_5[key] = dict_class_5[key] / anzahl

            temp_list = dict_class_6[key]
            dict_class_6[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_6[key][0])
            dict_class_6[key] = np.sum(dict_class_6[key],axis=0)
            dict_class_6[key] = dict_class_6[key] / anzahl

            if self.dict['classes'] > 6:
              temp_list = dict_class_7[key]
              dict_class_7[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_7[key][0])
              dict_class_7[key] = np.sum(dict_class_7[key],axis=0)
              dict_class_7[key] = dict_class_7[key] / anzahl

              temp_list = dict_class_8[key]
              dict_class_8[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_8[key][0])
              dict_class_8[key] = np.sum(dict_class_8[key],axis=0)
              dict_class_8[key] = dict_class_8[key] / anzahl

              temp_list = dict_class_9[key]
              dict_class_9[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_9[key][0])
              dict_class_9[key] = np.sum(dict_class_9[key],axis=0)
              dict_class_9[key] = dict_class_9[key] / anzahl

              temp_list = dict_class_10[key]
              dict_class_10[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_10[key][0])
              dict_class_10[key] = np.sum(dict_class_10[key],axis=0)
              dict_class_10[key] = dict_class_10[key] / anzahl
                                                  
          acc,loss = self.__test(epoch,test)
          l = []
          
          if self.dict['classes'] > 6: #Intel nur 6 Klassen deshalb differenzierung nötig
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6,dict_class_7,dict_class_8,dict_class_9,dict_class_10]
          else:
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6]
          
          row = (len(dict_class_1.keys()))  #Berechung wieviele Reihen für Plot
          
          for idx,n in enumerate(l):
              plot_dict = {
              'row' : int(row), #int da fehler bei float
              'col' : 1,
              'bins': 200,
              'label': f'Gradient distribution for class {idx+1}',
              'type' : 'hist',
              'act_fn': self.dict['act_fn'],
              'lr' : self.dict['lr'],
              'class' : {idx+1}
              }
              plot(n,plot_dict,epoch)

              for key in n.keys():
                n[key] = []

  def get_gradients_hook_in_out(self,train,test):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      global gradients_in
      gradients.clear() #reset des Dictionary
      gradients_in.clear()

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      set_backward_hook_in_out(self.model)
      for epoch in range(self.dict['epochs']):
        gradients.clear()  
        gradients_in.clear()
        acc = 0.0
        loss = 0.0
        print(f'Epoch: [{epoch+1} / {n_epochs}]')
        acc,loss = self.__train_grad(epoch,train)  
  
        with torch.no_grad():
            accs, loss = self.__test(epoch,test)

  
  def compute_gradients_per_class_hook_in_out(self,train,test,patience,akt):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      global gradients_in
      gradients.clear() #reset des Dictionary
      gradients_in.clear()
      dict_class_1 = {}
      dict_class_2 = {}
      dict_class_3 = {}
      dict_class_4 = {}
      dict_class_5 = {}
      dict_class_6 = {}
      if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
          dict_class_7 = {}
          dict_class_8 = {}
          dict_class_9 = {}
          dict_class_10 = {}

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      set_backward_hook_in_all_layers(self.model)
      set_backward_hook_out_all_layers(self.model)
      
      if akt == 0:
        k = ['ReLU()_6', 'ReLU()_5', 'ReLU()_4', 'ReLU()_3', 'ReLU()_2', 'ReLU()_1', 'ReLU()_0']
      if akt == 1:
        k = ['SiLU()_6', 'SiLU()_5', 'SiLU()_4', 'SiLU()_3', 'SiLU()_2', 'SiLU()_1', 'SiLU()_0']
      if akt == 2:
        k = ['Tanh()_6', 'Tanh()_5', 'Tanh()_4', 'Tanh()_3', 'Tanh()_2', 'Tanh()_1', 'Tanh()_0']
      for key in k:
          dict_class_1[key] = []
          dict_class_2[key] = []
          dict_class_3[key] = []
          dict_class_4[key] = []
          dict_class_5[key] = []
          dict_class_6[key] = []
          if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
              dict_class_7[key] = []
              dict_class_8[key] = []
              dict_class_9[key] = []
              dict_class_10[key] = []       
      
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

              t = 0 #Benötigt für Klassenaufteilung
              for i,idx in enumerate(targets[0]):
                  if idx == 1:
                      t = i #Ermittlung aktuellen Labels des Trainingsdatums 
              
              for key in gradients.keys():
                if t == 0:
                    dict_class_1[key].append(gradients[key])
                if t == 1:  
                    dict_class_2[key].append(gradients[key])
                if t == 2:
                    dict_class_3[key].append(gradients[key])
                if t == 3:
                    dict_class_4[key].append(gradients[key])
                if t == 4:
                    dict_class_5[key].append(gradients[key])
                if t == 5:
                    dict_class_6[key].append(gradients[key])
                  
                if self.dict['classes'] > 6:
                    if t == 6:
                        dict_class_7[key].append(gradients[key])
                    if t == 7:
                        dict_class_8[key].append(gradients[key])
                    if t == 8:
                        dict_class_9[key].append(gradients[key])
                    if t == 9:
                        dict_class_10[key].append(gradients[key]) 
          
          for key in dict_class_1.keys():
            temp_list = dict_class_1[key]
            dict_class_1[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_1[key][0])
            dict_class_1[key] = np.sum(dict_class_1[key],axis=0)
            dict_class_1[key] = dict_class_1[key] / anzahl

            temp_list = dict_class_2[key]
            dict_class_2[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_2[key][0])
            dict_class_2[key] = np.sum(dict_class_2[key],axis=0)
            dict_class_2[key] = dict_class_2[key] / anzahl

            temp_list = dict_class_3[key]
            dict_class_3[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_3[key][0])
            dict_class_3[key] = np.sum(dict_class_3[key],axis=0)
            dict_class_3[key] = dict_class_3[key] / anzahl

            temp_list = dict_class_4[key]
            dict_class_4[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_4[key][0])
            dict_class_4[key] = np.sum(dict_class_4[key],axis=0)
            dict_class_4[key] = dict_class_4[key] / anzahl

            temp_list = dict_class_5[key]
            dict_class_5[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_5[key][0])
            dict_class_5[key] = np.sum(dict_class_5[key],axis=0)
            dict_class_5[key] = dict_class_5[key] / anzahl

            temp_list = dict_class_6[key]
            dict_class_6[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_6[key][0])
            dict_class_6[key] = np.sum(dict_class_6[key],axis=0)
            dict_class_6[key] = dict_class_6[key] / anzahl

            if self.dict['classes'] > 6:
              temp_list = dict_class_7[key]
              dict_class_7[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_7[key][0])
              dict_class_7[key] = np.sum(dict_class_7[key],axis=0)
              dict_class_7[key] = dict_class_7[key] / anzahl

              temp_list = dict_class_8[key]
              dict_class_8[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_8[key][0])
              dict_class_8[key] = np.sum(dict_class_8[key],axis=0)
              dict_class_8[key] = dict_class_8[key] / anzahl

              temp_list = dict_class_9[key]
              dict_class_9[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_9[key][0])
              dict_class_9[key] = np.sum(dict_class_9[key],axis=0)
              dict_class_9[key] = dict_class_9[key] / anzahl

              temp_list = dict_class_10[key]
              dict_class_10[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_10[key][0])
              dict_class_10[key] = np.sum(dict_class_10[key],axis=0)
              dict_class_10[key] = dict_class_10[key] / anzahl
                                                  
          acc,loss = self.__test(epoch,test)
          l = []
          
          if self.dict['classes'] > 6: #Intel nur 6 Klassen deshalb differenzierung nötig
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6,dict_class_7,dict_class_8,dict_class_9,dict_class_10]
          else:
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6]
          
          row = (len(dict_class_1.keys()))  #Berechung wieviele Reihen für Plot
          
          for idx,n in enumerate(l):
              plot_dict = {
              'row' : int(row), #int da fehler bei float
              'col' : 1,
              'bins': 200,
              'label': f'Gradient distribution for class {idx+1}',
              'type' : 'hist',
              'act_fn': self.dict['act_fn'],
              'lr' : self.dict['lr'],
              'class' : {idx+1}
              }
              plot(n,plot_dict,epoch)

              for key in n.keys():
                n[key] = []
  
  
  def compute_gradients_per_class_hook_in_out_all(self,train,test):
      global gradients  #Dictionary aus helper datei mit aktuellen aktivierungswerten
      global gradients_in
      gradients.clear() #reset des Dictionary
      gradients_in.clear()
      dict_class_1 = {}
      dict_class_2 = {}
      dict_class_3 = {}
      dict_class_4 = {}
      dict_class_5 = {}
      dict_class_6 = {}
      if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
          dict_class_7 = {}
          dict_class_8 = {}
          dict_class_9 = {}
          dict_class_10 = {}

      dict_class_1_out = {}
      dict_class_2_out = {}
      dict_class_3_out = {}
      dict_class_4_out = {}
      dict_class_5_out = {}
      dict_class_6_out = {}
      if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
          dict_class_7_out = {}
          dict_class_8_out = {}
          dict_class_9_out = {}
          dict_class_10_out = {}

      self.model.to(self.device)
      n_epochs = self.dict['epochs']
      
      
      k = get_layer_names(self.model)

      for key in k:
          dict_class_1[key] = []
          dict_class_2[key] = []
          dict_class_3[key] = []
          dict_class_4[key] = []
          dict_class_5[key] = []
          dict_class_6[key] = []
          if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
              dict_class_7[key] = []
              dict_class_8[key] = []
              dict_class_9[key] = []
              dict_class_10[key] = []

      for key in k:
          if 'conv1' not in key:
            dict_class_1_out[key] = []
            dict_class_2_out[key] = []
            dict_class_3_out[key] = []
            dict_class_4_out[key] = []
            dict_class_5_out[key] = []
            dict_class_6_out[key] = []
            if self.dict['classes'] > 6:  #Intel hat nur 6 Klassen deshalb restriktion
                dict_class_7_out[key] = []
                dict_class_8_out[key] = []
                dict_class_9_out[key] = []
                dict_class_10_out[key] = []    

      set_backward_hook_in_out_all_layers(self.model)
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

              t = 0 #Benötigt für Klassenaufteilung
              for i,idx in enumerate(targets[0]):
                  if idx == 1:
                      t = i #Ermittlung aktuellen Labels des Trainingsdatums 
              
              for key in gradients.keys():
                if t == 0:
                    dict_class_1[key].append(gradients[key])
                    if 'conv1' not in key:
                      dict_class_1_out[key].append(gradients_in[key])
                if t == 1:  
                    dict_class_2[key].append(gradients[key])
                    if 'conv1' not in key:
                      dict_class_2_out[key].append(gradients_in[key])
                if t == 2:
                    dict_class_3[key].append(gradients[key])
                    if 'conv1' not in key:
                      dict_class_3_out[key].append(gradients_in[key])
                if t == 3:
                    dict_class_4[key].append(gradients[key])
                    if 'conv1' not in key:
                      dict_class_4_out[key].append(gradients_in[key])
                if t == 4:
                    dict_class_5[key].append(gradients[key])
                    if 'conv1' not in key:
                      dict_class_5_out[key].append(gradients_in[key])
                if t == 5:
                    dict_class_6[key].append(gradients[key])
                    if 'conv1' not in key:
                      dict_class_6_out[key].append(gradients_in[key])
                  
                if self.dict['classes'] > 6:
                    if t == 6:
                        dict_class_7[key].append(gradients[key])
                        if 'conv1' not in key:
                          dict_class_7_out[key].append(gradients[key])
                    if t == 7:
                        dict_class_8[key].append(gradients[key])
                        if 'conv1' not in key:
                          dict_class_8_out[key].append(gradients[key])
                    if t == 8:
                        dict_class_9[key].append(gradients[key])
                        if 'conv1' not in key:
                          dict_class_9_out[key].append(gradients[key])
                    if t == 9:
                        dict_class_10[key].append(gradients[key])
                        if 'conv1' not in key:
                          dict_class_10_out[key].append(gradients[key])
                                                                    
          for key in dict_class_1.keys():
            temp_list = dict_class_1[key]
            dict_class_1[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_1[key][0])
            dict_class_1[key] = np.sum(dict_class_1[key],axis=0)
            dict_class_1[key] = dict_class_1[key] / anzahl

            temp_list = dict_class_2[key]
            dict_class_2[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_2[key][0])
            dict_class_2[key] = np.sum(dict_class_2[key],axis=0)
            dict_class_2[key] = dict_class_2[key] / anzahl

            temp_list = dict_class_3[key]
            dict_class_3[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_3[key][0])
            dict_class_3[key] = np.sum(dict_class_3[key],axis=0)
            dict_class_3[key] = dict_class_3[key] / anzahl

            temp_list = dict_class_4[key]
            dict_class_4[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_4[key][0])
            dict_class_4[key] = np.sum(dict_class_4[key],axis=0)
            dict_class_4[key] = dict_class_4[key] / anzahl

            temp_list = dict_class_5[key]
            dict_class_5[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_5[key][0])
            dict_class_5[key] = np.sum(dict_class_5[key],axis=0)
            dict_class_5[key] = dict_class_5[key] / anzahl

            temp_list = dict_class_6[key]
            dict_class_6[key] = np.asarray(temp_list)
            anzahl = np.shape(dict_class_6[key][0])
            dict_class_6[key] = np.sum(dict_class_6[key],axis=0)
            dict_class_6[key] = dict_class_6[key] / anzahl

            if self.dict['classes'] > 6:
              temp_list = dict_class_7[key]
              dict_class_7[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_7[key][0])
              dict_class_7[key] = np.sum(dict_class_7[key],axis=0)
              dict_class_7[key] = dict_class_7[key] / anzahl

              temp_list = dict_class_8[key]
              dict_class_8[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_8[key][0])
              dict_class_8[key] = np.sum(dict_class_8[key],axis=0)
              dict_class_8[key] = dict_class_8[key] / anzahl

              temp_list = dict_class_9[key]
              dict_class_9[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_9[key][0])
              dict_class_9[key] = np.sum(dict_class_9[key],axis=0)
              dict_class_9[key] = dict_class_9[key] / anzahl

              temp_list = dict_class_10[key]
              dict_class_10[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_10[key][0])
              dict_class_10[key] = np.sum(dict_class_10[key],axis=0)
              dict_class_10[key] = dict_class_10[key] / anzahl
                      
            if 'conv1' not in key:
              temp_list = dict_class_1_out[key]
              dict_class_1_out[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_1_out[key][0])
              dict_class_1_out[key] = np.sum(dict_class_1_out[key],axis=0)
              dict_class_1_out[key] = dict_class_1_out[key] / anzahl

              temp_list = dict_class_2_out[key]
              dict_class_2_out[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_2_out[key][0])
              dict_class_2_out[key] = np.sum(dict_class_2_out[key],axis=0)
              dict_class_2_out[key] = dict_class_2_out[key] / anzahl

              temp_list = dict_class_3_out[key]
              dict_class_3_out[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_3_out[key][0])
              dict_class_3_out[key] = np.sum(dict_class_3_out[key],axis=0)
              dict_class_3_out[key] = dict_class_3_out[key] / anzahl

              temp_list = dict_class_4_out[key]
              dict_class_4_out[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_4_out[key][0])
              dict_class_4_out[key] = np.sum(dict_class_4_out[key],axis=0)
              dict_class_4_out[key] = dict_class_4_out[key] / anzahl

              temp_list = dict_class_5_out[key]
              dict_class_5_out[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_5_out[key][0])
              dict_class_5_out[key] = np.sum(dict_class_5_out[key],axis=0)
              dict_class_5_out[key] = dict_class_5_out[key] / anzahl

              temp_list = dict_class_6_out[key]
              dict_class_6_out[key] = np.asarray(temp_list)
              anzahl = np.shape(dict_class_6_out[key][0])
              dict_class_6_out[key] = np.sum(dict_class_6_out[key],axis=0)
              dict_class_6_out[key] = dict_class_6_out[key] / anzahl

              if self.dict['classes'] > 6:
                temp_list = dict_class_7_out[key]
                dict_class_7_out[key] = np.asarray(temp_list)
                anzahl = np.shape(dict_class_7_out[key][0])
                dict_class_7_out[key] = np.sum(dict_class_7_out[key],axis=0)
                dict_class_7_out[key] = dict_class_7_out[key] / anzahl

                temp_list = dict_class_8_out[key]
                dict_class_8_out[key] = np.asarray(temp_list)
                anzahl = np.shape(dict_class_8_out[key][0])
                dict_class_8_out[key] = np.sum(dict_class_8_out[key],axis=0)
                dict_class_8_out[key] = dict_class_8_out[key] / anzahl

                temp_list = dict_class_9_out[key]
                dict_class_9_out[key] = np.asarray(temp_list)
                anzahl = np.shape(dict_class_9_out[key][0])
                dict_class_9_out[key] = np.sum(dict_class_9_out[key],axis=0)
                dict_class_9_out[key] = dict_class_9_out[key] / anzahl

                temp_list = dict_class_10_out[key]
                dict_class_10_out[key] = np.asarray(temp_list)
                anzahl = np.shape(dict_class_10_out[key][0])
                dict_class_10_out[key] = np.sum(dict_class_10_out[key],axis=0)
                dict_class_10_out[key] = dict_class_10_out[key] / anzahl                                        
          
          acc,loss = self.__test(epoch,test)
          l = []
          
          if self.dict['classes'] > 6: #Intel nur 6 Klassen deshalb differenzierung nötig
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6,dict_class_7,dict_class_8,dict_class_9,dict_class_10]
          else:
            l = [dict_class_1,dict_class_2,dict_class_3,dict_class_4,dict_class_5,dict_class_6]
          
          row = (len(dict_class_1.keys()))  #Berechung wieviele Reihen für Plot
          
          for idx,n in enumerate(l):
              plot_dict = {
              'row' : int(row), #int da fehler bei float
              'col' : 1,
              'bins': 200,
              'label': f'Gradient distribution for class {idx+1}',
              'type' : 'hist',
              'act_fn': self.dict['act_fn'],
              'lr' : self.dict['lr'],
              'class' : {idx+1}
              }
              plot(n,plot_dict,epoch)

              for key in n.keys():
                n[key] = []
          
          if self.dict['classes'] > 6: #Intel nur 6 Klassen deshalb differenzierung nötig
            l = [dict_class_1_out,dict_class_2_out,dict_class_3_out,dict_class_4_out,dict_class_5_out,dict_class_6_out,dict_class_7_out,dict_class_8_out,dict_class_9_out,dict_class_10_out]
          else:
            l = [dict_class_1_out,dict_class_2_out,dict_class_3_out,dict_class_4_out,dict_class_5_out,dict_class_6_out]
          
          row = (len(dict_class_1_out.keys()))  #Berechung wieviele Reihen für Plot
          print('out')
          for idx,n in enumerate(l):
              i = idx+1
              cl = f'{i}_in'
              plot_dict = {
              'row' : int(row), #int da fehler bei float
              'col' : 1,
              'bins': 200,
              'label': f'Gradient distribution for class out {idx+1}',
              'type' : 'hist',
              'act_fn': self.dict['act_fn'],
              'lr' : self.dict['lr'],
              'class' : cl
              }
              plot(n,plot_dict,epoch)

              for key in n.keys():
                n[key] = []