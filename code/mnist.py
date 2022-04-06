from datasets.data import MNIST
from datasets.datasets import MnistDataset

from models.mnist_models import CNN_MNIST

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
batch_size = 128
n_epochs = 10
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
models = CNN_MNIST()


optimizer = optim.SGD(models.parameters(),lr=5e-3,momentum=0.9)
criterion = nn.CrossEntropyLoss()

n_epochs = 20

step_train = 0
step_test = 0

def evaluate(X, y, train=False):
    if train:
        models.zero_grad()
        
    scores = models(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(scores,y)]
    acc = matches.count(True)/len(matches)

    loss = criterion(scores,y)
    if train:
        loss.backward()
        optimizer.step()
    return acc,loss.item()


def train():
    models.train()
    acc = 0.0
    loss = 0.0
    step_train = 0
    for idx,(data,targets) in enumerate(train_l):
            
        data = data.to(device=device)
        targets = targets.to(device=device)

        acc, loss = evaluate(data,targets,train=True)

        if idx % 100:
            writer.add_scalar("Training loss", loss, global_step=step_train)
            writer.add_scalar("Training Accuracy", acc, global_step=step_train)
        step_train +=1
    mean_acc = acc / len(train_l.dataset)
    mean_loss = loss / len(train_l.dataset)
    writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
    writer.add_scalar('Mean Loss Train',mean_loss,epoch)
    
def test():
    step_test = 0
    with torch.no_grad():
        for idx,(data,targets) in enumerate(test_l):
            data = data.to(device=device)
            targets = targets.to(device=device)

            acc, loss = evaluate(data,targets,train=False)
            print(f'{loss} loss train')
            if idx % 100:
                writer.add_scalar("Test loss", loss, global_step=step_test)
                writer.add_scalar("Test Accuracy", acc, global_step=step_test)
            step_test +=1
    mean_acc = acc / len(test_l.dataset)
    mean_loss = loss / len(test_l.dataset)
    writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
    writer.add_scalar('Mean Loss Train',mean_loss,epoch)

for epoch in range(n_epochs):

    print(epoch)
    train()

    test()
