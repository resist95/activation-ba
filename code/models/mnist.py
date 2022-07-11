
import torch.nn as nn
import torch.nn.functional as F
import math
class CNN_MNIST_RELU(nn.Module):
  def __init__(self):
    super(CNN_MNIST_RELU,self).__init__()
    
    self.conv1 = nn.Conv2d(1,8,3,1)
    self.bn1 = nn.BatchNorm2d(8)
    self.relu1 = nn.ReLU()
    
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.bn2 = nn.BatchNorm2d(16)
    self.relu2 = nn.ReLU()
    
    self.pool = nn.MaxPool2d(2,2)
    
    self.conv3 = nn.Conv2d(16,16,3,1)
    self.bn3 = nn.BatchNorm2d(16)
    self.relu3 = nn.ReLU()
    
    self.conv4 = nn.Conv2d(16,32,3,1)
    self.bn4 = nn.BatchNorm2d(32)
    self.relu4 = nn.ReLU()
    
    self.fc1 = nn.Linear(2048,128)
    self.relu5 = nn.ReLU()
    self.drop_fc = nn.Dropout(0.9)
    
    self.fc2 = nn.Linear(128,10)


  def forward(self,x):
    x = self.relu1(self.bn1(self.conv1(x)))
    x = self.relu2(self.bn2(self.conv2(x)))
    x = self.pool(x)
    x = self.relu3(self.bn3(self.conv3(x)))
    x = self.relu4(self.bn4(self.conv4(x)))
    x = x.reshape(x.shape[0],-1)
    x = self.drop_fc(self.relu5(self.fc1(x)))
    x = self.fc2(x)
    return x



class CNN_MNIST_SWISH(nn.Module):
  def __init__(self):
    super(CNN_MNIST_SWISH,self).__init__()
    
    self.conv1 = nn.Conv2d(1,8,3,1)
    self.bn1 = nn.BatchNorm2d(8)
    self.silu1 = nn.SiLU()
    
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.bn2 = nn.BatchNorm2d(16)
    self.silu2 = nn.SiLU()
    
    self.pool = nn.MaxPool2d(2,2)
    
    self.conv3 = nn.Conv2d(16,16,3,1)
    self.bn3 = nn.BatchNorm2d(16)
    self.silu3 = nn.SiLU()
    
    self.conv4 = nn.Conv2d(16,32,3,1)
    self.bn4 = nn.BatchNorm2d(32)
    self.silu4 = nn.SiLU()
    
    self.fc1 = nn.Linear(2048,128)
    self.silu5 = nn.SiLU()
    self.drop_fc = nn.Dropout(0.9)

    self.fc2 = nn.Linear(128,10)

  def forward(self,x):
    x = self.silu1(self.bn1(self.conv1(x)))
    x = self.silu2(self.bn2(self.conv2(x)))
    x = self.pool(x)
    x = self.silu3(self.bn3(self.conv3(x)))
    x = self.silu4(self.bn4(self.conv4(x)))
    x = x.reshape(x.shape[0],-1)
    x = self.drop_fc(self.silu5(self.fc1(x)))
    x = self.fc2(x)
    return x



class CNN_MNIST_TANH(nn.Module):
  def __init__(self):
    super(CNN_MNIST_TANH,self).__init__()
    
    self.conv1 = nn.Conv2d(1,8,3,1)
    self.bn1 = nn.BatchNorm2d(8)
    self.tanh1 = nn.Tanh()
    
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.bn2 = nn.BatchNorm2d(16)
    self.tanh2 = nn.Tanh()

    self.pool = nn.MaxPool2d(2,2)

    self.conv3 = nn.Conv2d(16,16,3,1)
    self.bn3 = nn.BatchNorm2d(16)
    self.tanh3 = nn.Tanh()
    
    self.conv4 = nn.Conv2d(16,32,3,1)
    self.bn4 = nn.BatchNorm2d(32)
    self.tanh4 = nn.Tanh()
   
    self.fc1 = nn.Linear(2048,128)
    self.tanh5 = nn.Tanh()
    self.drop_fc = nn.Dropout(0.9)

    self.fc2 = nn.Linear(128,10)

  def forward(self,x):
    x = self.tanh1(self.bn1(self.conv1(x)))
    x = self.tanh2(self.bn2(self.conv2(x)))
    x = self.pool(x)
    x = self.tanh3(self.bn3(self.conv3(x)))
    x = self.tanh4(self.bn4(self.conv4(x)))
    x = x.reshape(x.shape[0],-1)
    x = self.drop_fc(self.tanh5(self.fc1(x)))
    x = self.fc2(x)
    return x

#Klassen für drop scheduler

class CNN_MNIST_RELU_drop_sched_0(nn.Module):
  def __init__(self):
    super(CNN_MNIST_RELU_drop_sched_0,self).__init__()
    self.relu = nn.ReLU()
    self.conv1 = nn.Conv2d(1,8,3,1)
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.conv3 = nn.Conv2d(16,16,3,1)
    self.conv4 = nn.Conv2d(16,32,3,1)
    self.pool = nn.MaxPool2d(2,2)

    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(16)
    self.bn3 = nn.BatchNorm2d(16)
    self.bn4 = nn.BatchNorm2d(32)

    self.drop1 = nn.Dropout(0.0)
    self.drop_fc = nn.Dropout(0.5)

    self.fc1 = nn.Linear(2048,128)
    self.fc2 = nn.Linear(128,10)
  
  def update(self,d):
    self.drop_fc = nn.Dropout(d)
    return d
  
  def forward(self,x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))

    x = self.pool(x)
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))
    x = x.reshape(x.shape[0],-1)
    x = self.drop_fc(self.relu(self.fc1(x)))
    x = self.fc2(x)
    return x

class CNN_MNIST_SWISH_drop_sched_0(nn.Module):
  def __init__(self):
    super(CNN_MNIST_SWISH_drop_sched_0,self).__init__()
    self.silu = nn.SiLU()
    self.conv1 = nn.Conv2d(1,8,3,1)
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.conv3 = nn.Conv2d(16,16,3,1)
    self.conv4 = nn.Conv2d(16,32,3,1)
    self.pool = nn.MaxPool2d(2,2)

    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(16)
    self.bn3 = nn.BatchNorm2d(16)
    self.bn4 = nn.BatchNorm2d(32)


    self.drop1 = nn.Dropout(0.0)
    self.drop_fc = nn.Dropout(0.5)

    self.fc1 = nn.Linear(2048,128)
    self.fc2 = nn.Linear(128,10)
  
  def update(self,d):
    self.drop_fc = nn.Dropout(d)
    return d

  def forward(self,x):
    x = self.silu(self.bn1(self.conv1(x)))
    x = self.silu(self.bn2(self.conv2(x)))

    x = self.pool(x)
    x = self.silu(self.bn3(self.conv3(x)))
    x = self.silu(self.bn4(self.conv4(x)))
    x = x.reshape(x.shape[0],-1)
    x = self.drop_fc(self.silu(self.fc1(x)))
    x = self.fc2(x)
    return x

class CNN_MNIST_TANH_drop_sched_0(nn.Module):
  def __init__(self):
    super(CNN_MNIST_TANH_drop_sched_0,self).__init__()
    self.tanh = nn.Tanh()
    self.conv1 = nn.Conv2d(1,8,3,1)
    self.conv2 = nn.Conv2d(8,16,3,1)
    self.conv3 = nn.Conv2d(16,16,3,1)
    self.conv4 = nn.Conv2d(16,32,3,1)
    self.pool = nn.MaxPool2d(2,2)

    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(16)
    self.bn3 = nn.BatchNorm2d(16)
    self.bn4 = nn.BatchNorm2d(32)

    self.drop1 = nn.Dropout(0.0)
    self.drop_fc = nn.Dropout(0.5)

    self.fc1 = nn.Linear(2048,128)
    self.fc2 = nn.Linear(128,10)
  
  def update(self,d):
    self.drop_fc = nn.Dropout(d)
    return d
  
  def forward(self,x):
    x = self.tanh(self.bn1(self.conv1(x)))
    x = self.tanh(self.bn2(self.conv2(x)))

    x = self.pool(x)
    x = self.tanh(self.bn3(self.conv3(x)))
    x = self.tanh(self.bn4(self.conv4(x)))
    x = x.reshape(x.shape[0],-1)
    x = self.drop_fc(self.tanh(self.fc1(x)))
    x = self.fc2(x)
    return x
from torchsummary import summary

model = CNN_MNIST_TANH()
model.to('cuda')
summary(model,(1,28,28))