import torch.nn as nn
import torch.nn.functional as F
'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,5,2) 
        self.fc1 = nn.Linear(78400,128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)


        return x'''

'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3,2)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2) 
        self.fc1 = nn.Linear(39936,128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)

        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x'''

'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3,2)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2) 
        self.fc1 = nn.Linear(7644,128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x'''


class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),3)
        self.conv3 = nn.Conv2d(96,156,(5,5),2) 
        self.bn3 = nn.BatchNorm2d(156)
        self.fc1 = nn.Linear(2496,128)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool1(x)

        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

from torchsummary import summary
model = CNN_INTEL(nn.ReLU,'relu')
summary(model, (3, 150, 150))