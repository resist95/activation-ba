from data import MNIST
from datasets import MnistDataset
from cnn import CNN_MNIST

import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchviz 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
x = current_time.replace(':','_')

writer = SummaryWriter(f'runs/mnist_{x}')

#device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#define param here
val_ratio = 0.1
batch_size = 64
n_epochs = 20
lr = 0.001

#load data
print('Loading Data... \n')
data = MNIST()

print('Done.')

#load into dataloader
dataset = MnistDataset

print('Loading train and test samples into DataLoader... \n')
X_train,y_train,X_test,y_test = data.get_data()
train = dataset(X_train,y_train)
test = dataset(X_test,y_test)
train_l = torch.utils.data.DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
test_l = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size,shuffle=False)

train_size = len(X_train)
test_size = len(X_test)

print('Done')

#declare nn here
mnist_cnn = CNN_MNIST()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mnist_cnn.parameters(),lr=lr,momentum=0.9)

models = mnist_cnn.to(device)

step_train = 0
step_test = 0

for epoch in range(n_epochs):
    #scheduler param
    losses = []
    #training and validation loss per epoch
    train_losses = 0.0
    valid_loss = 0.0

    train_acc = 0.0
    valid_acc = 0.0

    #Set model to train
    models.train()

    for batch_idx, (data,targets) in enumerate(train_l):
        
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        #zero gradients
        optimizer.zero_grad()

        #forward+backward
        scores = models(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        
        train_losses += loss.item() * data.size(0)

        _, preds = scores.max(1)
        correct_tensor = preds.eq(targets.data.view_as(preds))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

        train_acc += accuracy.item() * data.size(0)
        
        num_correct = (preds==targets).sum()
        running_train_acc = float(num_correct) /float(data.shape[0])

        writer.add_scalar("Training loss", loss, global_step=step_train)
        writer.add_scalar("Training Accuracy", running_train_acc, global_step=step_train)
        step_train +=1
    
    mean_loss = sum(losses) /len(losses)
    #scheduler.step(mean_loss)
    print(f"Cost at epoch {epoch} is {mean_loss}")
    #Set model to eval
    models.eval()

    with torch.no_grad():
        for data,targets in test_l:
            data = data.to(device)
            targets = targets.to(device)

            #forward
            scores = models(data)

            #validation loss
            loss = criterion(scores,targets)
            valid_loss += loss.item() * data.size(0)

            #validation acc
            _, preds = scores.max(1)
            correct_tensor = preds.eq(targets.data.view_as(preds))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor)
            )
            valid_acc += accuracy.item() * data.size(0)
            
            num_correct = (preds==targets).sum()
            running_test_acc = float(num_correct) /float(data.shape[0])

            writer.add_scalar("Validation loss", loss, global_step=step_test)
            writer.add_scalar("Validation Accuracy", running_test_acc, global_step=step_test)
            step_test +=1

    #Calculate average losses
    train_losses = train_losses /len(train_l.dataset)
    valid_losses = valid_loss / len(test_l.dataset)

    #Calculate average acc
    train_acc = train_acc /len(train_l.dataset)
    valid_acc = valid_acc /len(test_l.dataset)
    writer.add_scalar("Mean Loss Train", train_losses, global_step=epoch)
    writer.add_scalar("Mean Acc Train", train_acc, global_step=epoch)
    writer.add_scalar("Mean Loss Validation", valid_losses, global_step=epoch)
    writer.add_scalar("Mean Accuracy Test", valid_acc, global_step=epoch)
            
            

    print(f'Train Loss: {train_losses} Train Accuracy: {train_acc}')
    print(f'Valid Loss: {valid_losses} Valid Accuracy: {valid_acc}')
