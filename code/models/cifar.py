import torch.nn as nn
import torch.nn.functional as F


class CNN_CIFAR_RELU(nn.Module): 
    def __init__(self):
        super(CNN_CIFAR_RELU,self).__init__()
        #conv layers
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,5,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        
        self.avg = nn.AvgPool2d((5,5),1)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.drop = nn.Dropout(0.25)
        self.drop1 = nn.Dropout(0.1) 
        self.drop2 = nn.Dropout(0.05)
        self.drop3 = nn.Dropout(0.05)
        self.drop4 = nn.Dropout(0.05)
        self.drop5 = nn.Dropout(0.05)
        
        self.fc1 = nn.Linear(1024,128)
        self.fc2 = nn.Linear(128,10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop2(self.relu(self.bn3(self.conv3(x))))
        x = self.drop3(self.relu(self.bn4(self.conv4(x))))
        x = self.pool2(x)

        x = self.drop4(self.relu(self.bn5(self.conv5(x))))
        x = self.drop5(self.relu(self.bn6(self.conv6(x))))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN_CIFAR_SWISH(nn.Module): 
    def __init__(self):
        super(CNN_CIFAR_SWISH,self).__init__()
        #conv layers
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,5,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        
        self.avg = nn.AvgPool2d((3,3),1)
        self.pool1 = nn.MaxPool2d((2,2),2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.05)
        self.drop3 = nn.Dropout(0.1)
        self.drop4 = nn.Dropout(0.05)
        self.drop5 = nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(4096,128)
        self.fc2 = nn.Linear(128,10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
    
    def forward(self,x):
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.drop5(self.silu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop2(self.silu(self.bn3(self.conv3(x))))
        x = self.drop3(self.silu(self.bn4(self.conv4(x))))
        x = self.pool1(x)

        x = self.drop4(self.silu(self.bn5(self.conv5(x))))
        x = self.silu(self.bn6(self.conv6(x)))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop1(self.silu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN_CIFAR_TANH(nn.Module): 
    def __init__(self):
        super(CNN_CIFAR_TANH,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,5,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)

        self.drop_fc = nn.Dropout(0.1)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.0)
        self.drop3 = nn.Dropout(0.0)
        self.drop4 = nn.Dropout(0.0)
        self.drop5 = nn.Dropout(0.0)
        self.drop6 = nn.Dropout(0.0)

        self.avg = nn.AvgPool2d((5,5),1)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        #fc layers
        self.fc1 = nn.Linear(1024,128)
        self.fc2 = nn.Linear(128,10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,1)  
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,1)  
    def forward(self,x):
        x = self.drop1(F.tanh(self.bn1(self.conv1(x))))
        x = self.drop2(F.tanh(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop3(F.tanh(self.bn3(self.conv3(x))))
        x = self.drop4(F.tanh(self.bn4(self.conv4(x))))
        x = self.pool2(x)

        x = self.drop5(F.tanh(self.bn5(self.conv5(x))))
        x = self.drop6(F.tanh(self.bn6(self.conv6(x))))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(F.tanh(self.fc1(x)))
        x = self.fc2(x)
        return x