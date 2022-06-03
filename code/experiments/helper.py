#default dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sympy import N

#special dependencies
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sns.set()

#Vorbereitung Writer
def prepare_summary_writer(name,act_fn): #aktuell getestet variabel = name
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    x = current_time.replace(':','_') #benötigt um überschreibung zu vermeiden
    writer = SummaryWriter(f'runs/{name}_{act_fn}_{x}')
    return writer

#Vorbereitung GPU
def prepare_device():
    device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda:0':
        print('running on gpu')
    else:
        print('running on cpu')
    return device

#Plotten für eine Aktivierungsfunktion
def plot_single(train_acc,test_acc,train_loss,test_loss,plot_dict):
    i = np.min(train_acc) if np.min(train_acc) > np.min(test_acc) else np.min(test_acc)
    
    fig, axs = plt.subplots(2,figsize=(16,9),gridspec_kw={'height_ratios': [1, 2]})
    
    epochs = plot_dict['epochs']
    act_fn = plot_dict['act_fn']
    lr = plot_dict['lr']
    axs[0].plot(train_acc,label='Train Accuracy')
    axs[0].plot(test_acc,label='Test Accuracy')
    axs[0].set_xticks(np.arange(0, epochs,1))
    axs[0].set_yticks(np.arange(0.4,1,0.1))
    axs[0].set_title(f'Accuracy for Activation function {act_fn}')
    axs[1].plot(train_loss,label='Train Loss')
    axs[1].plot(test_loss,label='Test Loss')
    axs[1].set_xticks(np.arange(0,epochs,1))
    axs[1].set_yticks(np.arange(0,1.4,0.2))
    axs[1].set_title(f'Loss for Activation function {act_fn}')
    
    axs[0].legend()
    axs[1].legend()
  
    fig.tight_layout()
    plt.savefig(f'intel_{act_fn}_{lr}.pdf')
    plt.clf()

#Plot Histogramm und Line
#Histo: Gradient, Aktivierung
#Line: Accuracy, Loss für alle Aktivierungsfunktionen
def plot(data,plot_dict,epochs=0):
    act_fn = plot_dict['act_fn']
    num_plots = len(data.keys())  
    lr = plot_dict['lr']
    if plot_dict['type'] == 'hist':
        
        names = []
        for key in data.keys():
            names.append(key)
        i = 0
        if plot_dict['col'] >= 2:
          fig, axs = plt.subplots(int(plot_dict['row']),int(plot_dict['col']),figsize=(16,9))
          for row in range(plot_dict['row']):
              for col in range(plot_dict['col']):
                  if i < num_plots:
                      print(np.shape(data[names[i]]))
                      axs[row,col].hist(data[names[i]],
                                      bins=plot_dict['bins'],
                                      label=plot_dict['label'])
                      axs[row,col].set_title(names[i])
                  i +=1
          fig.tight_layout()
          classes = plot_dict['class']
          plt.savefig(f'cifar_act_{epochs}_{classes}_{act_fn}_{lr}.pdf')
          plt.clf()
          plt.cla()
          plt.close(fig)
        else: 
          fig, axs = plt.subplots(int(plot_dict['row']),figsize=(9,60)) 
          for row in range(plot_dict['row']):  
            
            axs[row].hist(data[names[row]],
                            bins=plot_dict['bins'],
                            label=plot_dict['label'])
            axs[row].set_title(names[row])
            fig.tight_layout()
            classes = plot_dict['class']
          plt.savefig(f'cifar_act_{epochs}_{row}_{classes}_{act_fn}_{lr}.pdf')
          plt.clf()
          plt.cla()
          plt.close(fig)
           
    elif plot_dict['type'] == 'line':
        fig, axs = plt.subplots(2)
        
        acc,loss = data
        
        names_acc = []
        for key in acc.keys():
            names_acc.append(key)
        
        names_loss = []
        
        for key in loss.keys():
            names_loss.append(key)
        
        max_acc = []
        max_loss = []
        for i in range (4):
            axs[0].plot((acc[names_acc[i]]),
                    label=f'{names_acc[i]} {lr}')
            max_acc.append(len(acc[names_acc[i]]))
            axs[1].plot((loss[names_loss[i]]),
                    label=f'{names_loss[i]} {lr}')
            max_loss.append(len(loss[names_loss[i]]))

        
        axs[0].set_title(f'Accuracy for Activation function {act_fn}')
        axs[1].set_title(f'Loss for Activation function {act_fn}')
        axs[0].set_xticks(np.arange(0, plot_dict['x_axis'],1))
        axs[0].set_yticks(np.arange(0,1,0.1))
        axs[1].set_xticks(np.arange(0,plot_dict['x_axis'],1))
        axs[1].set_yticks(np.arange(0,plot_dict['y_axis'],1))      
        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        classes = plot_dict['class']
        plt.savefig(f'acc_loss_{act_fn}_{lr}_fill.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig)
    
    elif plot_dict['type'] == 'acti':
        names = []
        for key in data.keys():
            names.append(key)
        
        counter_conv = 0
        counter_fc = 0
        counter_pool = 0
        counter_bn = 0
        counter_drop = 0
        for n in names:
          if 'conv' in n:
            counter_conv +=1
          elif 'fc' in n:
            counter_fc +=1
          elif 'pool' in n:
            counter_pool +=1
          elif 'bn' in n:
            counter_bn +=1
          elif 'drop' in n:
            counter_drop +=1
        
        #conv
        fig, axs = plt.subplots(counter_conv,figsize=(16,9))
        l = 0
        for i,n in enumerate(names):
          
          if 'conv' in n:
            axs[l].hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
            axs[l].set_title(names[i])
            l+= 1

        fig.tight_layout()
        classes = plot_dict['class']
        plt.savefig(f'cifar_act_{epochs}_{classes}_{act_fn}_{lr}_conv.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig)

        fig, axs = plt.subplots(counter_fc,figsize=(16,9))
        l = 0
        for i,n in enumerate(names):
          if 'fc' in n:
            axs[l].hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
            axs[l].set_title(names[i])
            l+= 1

        fig.tight_layout()
        classes = plot_dict['class']
        plt.savefig(f'cifar_act_{epochs}_{classes}_{act_fn}_{lr}_fc.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig)

        fig, axs = plt.subplots(counter_pool,figsize=(16,9))
        l = 0
        for i,n in enumerate(names):
          if 'pool' in n:
            if counter_pool > 1:
              axs[l].hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
              axs[l].set_title(names[i])
              l+= 1
            else:
              axs.hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
              axs.set_title(names[i])
              l+= 1

        fig.tight_layout()
        classes = plot_dict['class']
        plt.savefig(f'cifar_act_{epochs}_{classes}_{act_fn}_{lr}_pool.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig)

        fig, axs = plt.subplots(counter_bn,figsize=(16,9))
        l = 0
        for i,n in enumerate(names):
          if 'bn' in n:
            if counter_bn > 1:
              axs[l].hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
              axs[l].set_title(names[i])
              l+= 1
            else:
              axs.hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
              axs.set_title(names[i])
              l+= 1

        fig.tight_layout()
        classes = plot_dict['class']
        plt.savefig(f'cifar_act_{epochs}_{classes}_{act_fn}_{lr}_bn.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig)

        fig, axs = plt.subplots(((counter_drop+1)),figsize=(16,9))
        l = 0
        for i,n in enumerate(names):
          if 'drop' in n:
            if counter_drop > 1:
              axs[l].hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
              axs[l].set_title(names[i])
              l+= 1
            else:
              axs.hist(data[names[i]],
                             bins=plot_dict['bins'],
                             label=plot_dict['label'])
              axs.set_title(names[i])
              l+= 1

        fig.tight_layout()
        classes = plot_dict['class']
        plt.savefig(f'cifar_act_{epochs}_{classes}_{act_fn}_{lr}_drop.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig)


#Berechnung Aktivierungen pro Layer
activations = {}
def get_activations(name):
    def hook(module,input,output):
      activations[name] = output.view(-1).cpu().detach().numpy()
    return hook

def get_activations_feature_map(name):
    def hook(module,input,output):
      activations[name] = output.cpu().detach().numpy()
    return hook

def get_activations_input(name):
    def hook(module,input,output):

      activations[name] = input[0].view(-1).cpu().detach().numpy()
    return hook

def get_activations_feature_map_input(name):
    def hook(module,input,output):
      if 'conv' in name:
        activations[name] = input[0].cpu().detach().numpy()
    return hook

#Berechnung Gradienten pro Layer
def get_gradients(model,p):
    grad = {}
    for name, layer in model.named_children():
        if isinstance(layer,nn.Sequential): #rekursiv
            get_gradients(layer,p)
        else:
            layer_name = name
            if 'bn' not in layer_name:
                if 'pool'  not in layer_name : 
                    if 'drop' not in layer_name :
                        for name, param in layer.named_parameters():
                            if isinstance(layer,nn.ReLU):
                              if p in name:
                                    grad[layer_name] = param.view(-1).cpu().detach().numpy()
                            if isinstance(layer, nn.Linear):
                                if p in name:
                                    grad[layer_name] = param.view(-1).cpu().detach().numpy()
                            if isinstance(layer, nn.Conv2d):
                                if p in name:
                                    grad[layer_name] = param.view(-1).cpu().detach().numpy()
    return grad

#Namen der Layer des CNN
def get_layer_names(model):
    names = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            get_layer_names(layer)
        else:
            names.append(name)
    return names

#Hook setup
def set_hook(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_hook(layer)
    else:
      layer.register_forward_hook(get_activations(name))

def set_hook_feature_map(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_hook_feature_map(layer)
    else:
      layer.register_forward_hook(get_activations_feature_map(name))

def set_hook_in(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_hook(layer)
    else:
      layer.register_forward_hook(get_activations_input(name))

def set_hook_feature_map_in(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_hook_feature_map(layer)
    else:
      layer.register_forward_hook(get_activations_feature_map_input(name))




#backward#####

gradients = {}
gradients_in = {}
def get_grads_input(name,i):
    def hook(module,input,output):
      n = f'{name}_{i}'
      gradients_in[n] = input[0].view(-1).cpu().detach().numpy()
    return hook

def get_grads_output(name,i):
  def hook(module,input,output):
      n = f'{name}_{i}'
      gradients[n] = output[0].view(-1).cpu().detach().numpy()
  return hook

def set_backward_hook(model):
  i = 0
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_backward_hook(layer)
    else:
      if isinstance(layer, nn.ReLU):
        layer.register_full_backward_hook(get_grads_input(name,i))
        i = i + 1
      if isinstance(layer, nn.SiLU):
        layer.register_full_backward_hook(get_grads_input(name,i))
        i = i + 1
      if isinstance(layer, nn.Tanh):
        layer.register_full_backward_hook(get_grads_input(name,i))
        i = i + 1

def set_backward_hook_out(model):
  i = 0
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_backward_hook_out(layer)
    else:
      if isinstance(layer, nn.ReLU):
        layer.register_full_backward_hook(get_grads_output(name,i))
        i = i + 1
      if isinstance(layer, nn.SiLU):
        layer.register_full_backward_hook(get_grads_output(name,i))
        i = i + 1
      if isinstance(layer, nn.Tanh):
        layer.register_full_backward_hook(get_grads_output(name,i))
        i = i + 1

def set_backward_hook_in_out(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_backward_hook(layer)
    else:
      if isinstance(layer, nn.ReLU):
        layer.register_full_backward_hook(get_grads_in_out(name))

      if isinstance(layer, nn.SiLU):
        layer.register_full_backward_hook(get_grads_in_out(name))

      if isinstance(layer, nn.Tanh):
        layer.register_full_backward_hook(get_grads_in_out(name))

def get_grads_in_out(name):
  def hook(module,input,output):
      
      gradients[name] = output[0].view(-1).cpu().detach().numpy()
      if 'conv1' not in name:
        gradients_in[name] = input[0].view(-1).cpu().detach().numpy()
  return hook

def set_backward_hook_in_out_all_layers(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_backward_hook(layer)
    else:
      if isinstance(layer, nn.ReLU):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.SiLU):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.Tanh):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.Conv2d):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.MaxPool2d):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.BatchNorm2d):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.Dropout):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.Linear):
        layer.register_full_backward_hook(get_grads_in_out(name))
      if isinstance(layer, nn.AvgPool2d):
        layer.register_full_backward_hook(get_grads_in_out(name))
      

def set_backward_hook_out_all_layers(model):
  for name, layer in model.named_modules():
    if isinstance(layer,nn.Sequential):
      set_backward_hook_out(layer)
    else:
      if isinstance(layer, nn.ReLU):
        layer.register_full_backward_hook(get_grads_output(name,acti))
        acti = acti + 1
      if isinstance(layer, nn.SiLU):
        layer.register_full_backward_hook(get_grads_output(name,acti))
        acti = acti + 1
      if isinstance(layer, nn.Tanh):
        layer.register_full_backward_hook(get_grads_output(name,acti))
        acti = acti + 1
      
      if isinstance(layer, nn.Conv2d):
        layer.register_full_backward_hook(get_grads_output(name,conv))
        conv = conv + 1
      if isinstance(layer, nn.MaxPool2d):
        layer.register_full_backward_hook(get_grads_output(name,pool))
        pool = pool +1
      if isinstance(layer, nn.BatchNorm2d):
        layer.register_full_backward_hook(get_grads_output(name,bn))
        bn = bn +1
      if isinstance(layer, nn.Dropout):
        layer.register_full_backward_hook(get_grads_output(name,drop))
        drop = drop +1
      if isinstance(layer, nn.Linear):
        layer.register_full_backward_hook(get_grads_output(name,fc))
        fc = fc +1
      if isinstance(layer, nn.AvgPool2d):
        layer.register_full_backward_hook(get_grads_output(name,avg))
        avg = avg +1