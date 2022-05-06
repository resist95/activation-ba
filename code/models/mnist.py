
import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST_RELU(nn.Module):
  def __init__(self):
    super(CNN_MNIST_RELU,self).__init__()
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

    self.drop1 = nn.Dropout(0.1)
    self.drop_fc = nn.Dropout(0.25)

    self.fc1 = nn.Linear(2048,128)
    self.fc2 = nn.Linear(128,10)

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

class CNN_MNIST_SWISH(nn.Module):
  def __init__(self):
    super(CNN_MNIST_SWISH,self).__init__()
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


    self.drop1 = nn.Dropout(0.1)
    self.drop_fc = nn.Dropout(0.25)

    self.fc1 = nn.Linear(2048,128)
    self.fc2 = nn.Linear(128,10)

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

class CNN_MNIST_TANH(nn.Module):
  def __init__(self):
    super(CNN_MNIST_TANH,self).__init__()
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

    self.drop1 = nn.Dropout(0.1)
    self.drop_fc = nn.Dropout(0.25)

    self.fc1 = nn.Linear(2048,128)
    self.fc2 = nn.Linear(128,10)

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