
from data import CIFAR10
from data import caltech101
from data import MNIST
from data import Intel

from datasets import Caltech101Dataset
from datasets import Cifar10Dataset
from datasets import MnistDataset
from datasets import IntelDataset

from cnn import CNN_CIFAR
from tqdm import tqdm
import time

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchviz 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mnist')

def make_train_step(model,loss_fn,optimizer):
    def train_step(X,y):
        model.train()        
        pred = model(X)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

#device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#define param here
val_ratio = 0.1
batch_size = 100
n_epochs = 2
lr = 0.01


#load data
print('Loading Data... \n')
cif = CIFAR10()
cal = caltech101()
mni = MNIST()
inte = Intel()

data = [cif,mni,inte,cal]
data_names = ['MNIST_Dataset','CIFAR_Dataset','Intel_Dataset','Caltech_Dataset']

print('Done.')

#load into dataloader
mnis = MnistDataset
cifa = Cifar10Dataset
calt = Caltech101Dataset
intel = IntelDataset

datasets = [cifa,mnis,intel,calt]
train_loader = []
test_loader = []

print('Loading train and test samples into DataLoader... \n')
for i,dat in enumerate(data):
    X_train,y_train,X_test,y_test = dat.get_data()
    train = datasets[i](X_train,y_train)
    test = datasets[i](X_test,y_test)
    train_l = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    test_l = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
    train_loader.append(train_l)
    test_loader.append(test_l)

print('Done')

#declare nn here
mni_nn = 'define here'
cif_nn = CNN_CIFAR()
int_nn = 'define here'
cal_nn = 'define here'

#model = [mni_nn,cif_nn,int_nn,cal_nn]
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cif_nn.parameters(),lr=lr,momentum=0.9)   


for train,test in zip (train_loader[0],test_loader[0]):
    models = cif_nn.to(device)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(n_epochs):
        t_loss = 0.0
        t_acc = 0.0
        v_loss = 0.0
        v_acc = 0.0
        for (X_Train,y_train) in train:
            #train_loss,train_acc = train(models,X_train,y_train)
            models.train()

            X_train = X_train.to(device)
            y_train = y_train.to(device)
    
            optimizer.zero_grad()

            outputs = models(X_train)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs,y_test)

            loss.backward()
            optimizer.step()

            t_loss += loss.item() * X_train.size(0)
            t_acc += torch.sum(preds == y_test).item()
        
        print('| End of epoch: {:3d} |  Train loss: {:.2f} | Train acc: {:.2f}|'
              .format(epoch, t_loss, t_acc))  

        train_loss.append(t_loss)
        train_acc.append(t_acc)

        for X_test,y_test in test:

            models.eval()

            with torch.no_grad():

                X_test = X_test.to(device)
                y_test = y_test.to(device)
                outputs = models(X_test)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs,y_test)

                v_loss += loss.item() * X_test.size(0)
                v_acc += torch.sum(preds == y_test.data).item()
                
            val_loss.append(v_loss)
            val_acc.append(v_acc)

        print('| End of epoch: {:3d} |  Val loss: {:.2f} | Val acc: {:.2f}|'
              .format(epoch, v_loss, v_acc))




'''
#model = cif_nn.to(device)


def train(net,X_train,y_train):
    net.train()
    running_loss = 0.0
    running_corrects = 0.0

    train = X_train.to(device)
    labels = y_train.to(device)
    
    optimizer.zero_grad()

    outputs = net(train)
    _, preds = torch.max(outputs,1)
    loss = criterion(outputs,labels)

    loss.backward()
    optimizer.step()

    running_loss += loss.item() * X_train.size(0)
    running_corrects += torch.sum(preds == labels).item()

    return running_loss,running_corrects

    
def eval(net,X_test,y_test):
    net.eval()
    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():

        test = X_test.to(device)
        labels = y_test.to(device)
        outputs = net(test)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs,labels)

        running_loss += loss.item() * X_test.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
    return running_loss,running_corrects
#epochs,losses = train(model,train_loader)
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for (X_test,y_test) in test_loader[0]:
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        outputs = model(X_test)
        _, predicted = torch.max(outputs.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()
print(f'Accuracy of NN on {total} test images: {100*correct / total:.2f}%')



running_loss += loss.item() * batch_size

        print(f'[Epoch: {epoch+1}, Step: {i+1}] Loss: {running_loss/(i+1):.3f}')
        epoch_list.append(epoch)
        loss_list.append(running_loss)

            
    end = time.time()
    print('Done Training..')
    print(f'Training time: {(end-start)/60:.2f} Minutes')
    return epoch_list, loss_list



for train,test in zip(train_loader,test_loader):
    #define loss,optimizer
    #curr_model = cif_nn
    train_loss, train_accuracy = [], []
    curr_model = cif_nn.to(device)    
    
    #train_step = make_train_step(curr_model,loss_fn,optimizer)
    
    for epoch in range(n_epochs):
        print('Training')
        running_loss = 0.0
        running_correct = 0.0
        cif_nn.train()
        for i, (X_train,y_train) in enumerate(train):
            print(f'Epoch {epoch+1}/{n_epochs}')

            X_train = X_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            output = cif_nn(X_train)
            print(y_train.shape)
            print(output.shape)
            loss = criterion(output, y_train)
            running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            running_correct += (preds == torch.max(y_train, 1)[1].sum().item())
            
            loss.backward()
            optimizer.step()
            print(f'Loss: {loss.item()}')
        
        loss = running_loss/len(train.dataset)
        accuracy = 100* running_correct/len(train.dataset)
    
    
         
        curr_model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for X_test,y_test in test:
                y_val = curr_model(X_test)
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                
                _, predicted = torch.max(y_val.data, 1)
                n_samples += y_test.size(0)
                n_correct += (predicted == y_test).sum().item()

                val_loss = loss_fn(y_val,y_test)
                valid_loss += val_loss.item()*X_test.size(0)

            acc = 100.0 * n_correct /n_samples
            print(f'Accuracy of the network on the test images: {acc} %')
                #val_losses.append(val_loss.item())
        train_loss = train_loss/len(train.sampler)
        valid_loss = valid_loss/len(test.sampler)

        print(f'train loss over one Epoch: {train_loss} %')
        print(f'valid loss over one Epoch: {valid_loss} %')
    i +=1
          
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

i = 0
for train,test in zip(train_loader,test_loader):
    n_total_steps = len(train)
    for epoch in range(n_epochs):

        for i, X_train,y_train in enumerate(train):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            outputs = cif_nn(X_train)
            loss = criterion(outputs,y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
'''

