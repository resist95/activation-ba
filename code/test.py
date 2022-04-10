
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from context import Intel
from context import IntelDataset_random_mean
from cnn import CNN_INTEL

import torch.nn as nn
import torch.optim as optim

#Set seed for repoducability
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


#Setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running CNN on {device}')

now = datetime.now()

#Setup Tensorboard
current_time = now.strftime("%H:%M:%S")
x = current_time.replace(':','_')

writer = SummaryWriter(f'runs/intel_{x}')


intel = Intel()
inte = IntelDataset_random_mean

X_train,y_train,X_test,y_test = intel.get_data()

trains = inte(X_train,y_train,train=True)
tests = inte(X_test,y_test,train=False)
train_l = torch.utils.data.DataLoader(dataset=trains,batch_size=256,shuffle=True,num_workers = 0)
test_l = torch.utils.data.DataLoader(dataset=tests,batch_size=256,shuffle=False, num_workers = 0)

lr = 0.003

cif_nn = CNN_INTEL()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cif_nn.parameters(),lr=lr,momentum=0.9)

models = cif_nn.to(device)

step_train = 0
step_test = 0
n_epochs = 2

# Define Scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,20]
)

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
    
    means = torch.zeros(3)
    stds = torch.zeros(3)
    for idx,(data,targets) in enumerate(train_l):
        
        m, s = trains.get_mean_std()
        m /= data.size(0)
        s /= data.size(0)
        means += m
        stds += s
        data = data.to(device=device)
        targets = targets.to(device=device)
        accs, losses = evaluate(data,targets,train=True)
        loss += losses * data.size(0)
        acc += accs * data.size(0)
    
    mean_acc = acc / len(train_l.dataset)
    mean_loss = loss / len(train_l.dataset)
    writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
    writer.add_scalar('Mean Loss Train',mean_loss,epoch)
    return means,stds
    
def test(mean,std):
    acc = 0.0
    loss = 0.0

    tests.set_mean_std(mean,std)
    with torch.no_grad():
        for (data,targets) in test_l:
            data = data.to(device=device)
            targets = targets.to(device=device)

            accs, losses = evaluate(data,targets,train=False)
            loss += losses * data.size(0)
            acc += accs * data.size(0)
    mean_acc = acc / len(test_l.dataset)
    mean_loss = loss / len(test_l.dataset)
    writer.add_scalar('Mean Accuracy Train',mean_acc,epoch)
    writer.add_scalar('Mean Loss Train',mean_loss,epoch)

for epoch in range(n_epochs):
    means = torch.zeros(3)
    stds = torch.zeros(3)
    print(epoch)
    means, stds = train()
    means /= len(train_l)
    stds /= len(train_l)
    test(means,stds)
        

