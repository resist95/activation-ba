import torch.nn as nn
import torch.nn.functional as F
import math

class CNN_INTEL_RELU(nn.Module):
    def __init__(self):
        super(CNN_INTEL_RELU,self).__init__()
        
        

        self.conv1 = nn.Conv2d(3,96,5,3,1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.bn2 = nn.BatchNorm2d(96)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.0)
        
        self.pool1 = nn.MaxPool2d((3,3),2,1)
        
        self.conv3 = nn.Conv2d(96,128,(6,6),1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.0)
        
        self.conv4 = nn.Conv2d(128,128,(3,3),1,1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.0)
        
        self.pool2 = nn.MaxPool2d((3,3),2,1)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.0)
        
        self.conv6 = nn.Conv2d(256,256,3,2,1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.0)
                
        self.avg = nn.AvgPool2d((3,3),1)
        
        self.fc1 = nn.Linear(1024,128)
        self.relu7 = nn.ReLU()
        self.drop_fc = nn.Dropout(0.8)
        
        self.fc2 = nn.Linear(128,6)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  
    
    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.drop4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.drop5(self.relu5(self.bn5(self.conv5(x))))
        x = self.drop6(self.relu6(self.bn6(self.conv6(x))))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(self.relu7(self.fc1(x)))
        x = self.fc2(x)

        return x

class CNN_INTEL_SWISH(nn.Module):
    def __init__(self):
        super(CNN_INTEL_SWISH,self).__init__()
        
        self.conv1 = nn.Conv2d(3,96,5,3,1)
        self.bn1 = nn.BatchNorm2d(96)
        self.silu1 = nn.SiLU()
        self.drop2 = nn.Dropout(0.0)
        
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.bn2 = nn.BatchNorm2d(96)
        self.silu2 = nn.SiLU()
        
        self.pool1 = nn.MaxPool2d((3,3),2,1)
        
        self.conv3 = nn.Conv2d(96,128,(6,6),1)
        self.bn3 = nn.BatchNorm2d(128)
        self.silu3 = nn.SiLU()
        self.drop5 = nn.Dropout(0.0)

        self.conv4 = nn.Conv2d(128,128,(3,3),1,1)
        self.bn4 = nn.BatchNorm2d(128)
        self.silu4 = nn.SiLU()
        self.drop3 = nn.Dropout(0.0)
        
        self.pool2 = nn.MaxPool2d((3,3),2,1)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256)
        self.silu5 = nn.SiLU()
        self.drop4 = nn.Dropout(0.0)

        self.conv6 = nn.Conv2d(256,256,3,2,1)
        self.bn6 = nn.BatchNorm2d(256)
        self.silu6 = nn.SiLU()
        
        self.avg = nn.AvgPool2d((3,3),1)
        
        self.fc1 = nn.Linear(1024,128)
        self.silu7 = nn.SiLU()
        self.drop_fc = nn.Dropout(0.8)
        self.fc2 = nn.Linear(128,6)
             
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  
    
    def forward(self,x):
        x = self.drop2(self.silu1(self.bn1(self.conv1(x))))
        x = self.silu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop5(self.silu3(self.bn3(self.conv3(x))))
        x = self.drop3(self.silu4(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.drop4(self.silu5(self.bn5(self.conv5(x))))
        x = self.silu6(self.bn6(self.conv6(x)))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(self.silu7(self.fc1(x)))
        x = self.fc2(x)

        return x

class CNN_INTEL_TANH(nn.Module):
    def __init__(self):
        super(CNN_INTEL_TANH,self).__init__()
    
        self.conv1 = nn.Conv2d(3,96,5,3,1)
        self.bn1 = nn.BatchNorm2d(96)
        self.tanh1 = nn.Tanh()
        self.drop1 = nn.Dropout(0.0)
        
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.bn2 = nn.BatchNorm2d(96)
        self.tanh2 = nn.Tanh()
        self.drop2 = nn.Dropout(0.0)
        
        self.pool1 = nn.MaxPool2d((3,3),2,1)
        
        self.conv3 = nn.Conv2d(96,128,(6,6),1)
        self.bn3 = nn.BatchNorm2d(128)
        self.tanh3 = nn.Tanh()
        self.drop3 = nn.Dropout(0.0)

        self.conv4 = nn.Conv2d(128,128,(3,3),1,1)
        self.bn4 = nn.BatchNorm2d(128)
        self.tanh4 = nn.Tanh()
        self.drop4 = nn.Dropout(0.0)   
        
        self.pool2 = nn.MaxPool2d((3,3),2,1)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256)
        self.tanh5 = nn.Tanh()
        self.drop5 = nn.Dropout(0.0)

        self.conv6 = nn.Conv2d(256,256,3,2,1)
        self.bn6 = nn.BatchNorm2d(256)
        self.tanh6 = nn.Tanh()
        self.drop6 = nn.Dropout(0.0)
        
        self.avg = nn.AvgPool2d((3,3),1)
        
        self.fc1 = nn.Linear(2304,128)
        self.tanh7 = nn.Tanh()
        self.drop_fc = nn.Dropout(0.8)
        self.fc2 = nn.Linear(128,6)
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,1)  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,1)  
    
    def forward(self,x):
        x = self.drop1(self.tanh1(self.bn1(self.conv1(x))))
        x = self.drop2(self.tanh2(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop3(self.tanh3(self.bn3(self.conv3(x))))
        x = self.drop4(self.tanh4(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.drop5(self.tanh5(self.bn5(self.conv5(x))))
        x = self.drop6(self.tanh6(self.bn6(self.conv6(x))))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(self.tanh7(self.fc1(x)))
        x = self.fc2(x)

        return x


#Klassen für Dropout Scheduler

class CNN_INTEL_RELU_drop_sched_0(nn.Module):
    def __init__(self):
        super(CNN_INTEL_RELU_drop_sched_0,self).__init__()

        self.conv1 = nn.Conv2d(3,96,5,3,1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.bn2 = nn.BatchNorm2d(96)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.0)
        
        self.pool1 = nn.MaxPool2d((3,3),2,1)
        
        self.conv3 = nn.Conv2d(96,128,(6,6),1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.0)

        self.conv4 = nn.Conv2d(128,128,(3,3),1,1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.0)
        
        self.pool2 = nn.MaxPool2d((3,3),2,1)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.0)

        self.conv6 = nn.Conv2d(256,256,3,2,1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.0)
                
        self.avg = nn.AvgPool2d((3,3),1)
        
        self.fc1 = nn.Linear(2304,128)
        self.relu7 = nn.ReLU()
        self.drop_fc = nn.Dropout(0.0)
        
        self.fc2 = nn.Linear(128,6)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  
    
    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.drop4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.drop5(self.relu5(self.bn5(self.conv5(x))))
        x = self.drop6(self.relu6(self.bn6(self.conv6(x))))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(self.relu7(self.fc1(x)))
        x = self.fc2(x)

        return x  
    
    def update(self,d):
        self.drop_fc = nn.Dropout(d)
        return d
    
    def update_drop(self,i):
        drop_rate = math.log(i+3) /6
        
        if drop_rate < 0.9:
            self.drop_fc = nn.Dropout(drop_rate)
            return drop_rate
        else: 
            self.drop_fc = nn.Dropout(0.9)
            return 0.9
        

class CNN_INTEL_SWISH_drop_sched_0(nn.Module):
    def __init__(self):
        super(CNN_INTEL_SWISH_drop_sched_0,self).__init__()
        
        self.conv1 = nn.Conv2d(3,96,5,3,1)
        self.bn1 = nn.BatchNorm2d(96)
        self.silu1 = nn.SiLU()
        self.drop2 = nn.Dropout(0.0)
        
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.bn2 = nn.BatchNorm2d(96)
        self.silu2 = nn.SiLU()
        
        self.pool1 = nn.MaxPool2d((3,3),2,1)
        
        self.conv3 = nn.Conv2d(96,128,(6,6),1)
        self.bn3 = nn.BatchNorm2d(128)
        self.silu3 = nn.SiLU()
        self.drop5 = nn.Dropout(0.0)

        self.conv4 = nn.Conv2d(128,128,(3,3),1,1)
        self.bn4 = nn.BatchNorm2d(128)
        self.silu4 = nn.SiLU()
        self.drop3 = nn.Dropout(0.0)
        
        self.pool2 = nn.MaxPool2d((3,3),2,1)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256)
        self.silu5 = nn.SiLU()
        self.drop4 = nn.Dropout(0.0)

        self.conv6 = nn.Conv2d(256,256,3,2,1)
        self.bn6 = nn.BatchNorm2d(256)
        self.silu6 = nn.SiLU()
        
        self.avg = nn.AvgPool2d((3,3),1)
        
        self.fc1 = nn.Linear(2304,128)
        self.silu7 = nn.SiLU()
        self.drop_fc = nn.Dropout(0.0)
        self.fc2 = nn.Linear(128,6)
             
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  
    
    def forward(self,x):
        x = self.drop2(self.silu1(self.bn1(self.conv1(x))))
        x = self.silu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop5(self.silu3(self.bn3(self.conv3(x))))
        x = self.drop3(self.silu4(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.drop4(self.silu5(self.bn5(self.conv5(x))))
        x = self.silu6(self.bn6(self.conv6(x)))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(self.silu7(self.fc1(x)))
        x = self.fc2(x)

        return x  
    
    def update(self,d):
        self.drop_fc = nn.Dropout(d)
        return d
    
    def update_drop(self,i):
        drop_rate = math.log(i+3) /6
        
        if drop_rate < 0.9:
            self.drop_fc = nn.Dropout(drop_rate)
            return drop_rate
        else: 
            self.drop_fc = nn.Dropout(0.9)
            return 0.9


class CNN_INTEL_TANH_drop_sched_0(nn.Module):
    def __init__(self):
        super(CNN_INTEL_TANH_drop_sched_0,self).__init__()
    
        self.conv1 = nn.Conv2d(3,96,5,3,1)
        self.bn1 = nn.BatchNorm2d(96)
        self.tanh1 = nn.Tanh()
        self.drop1 = nn.Dropout(0.0)
        
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.bn2 = nn.BatchNorm2d(96)
        self.tanh2 = nn.Tanh()
        self.drop2 = nn.Dropout(0.0)
        
        self.pool1 = nn.MaxPool2d((3,3),2,1)
        
        self.conv3 = nn.Conv2d(96,128,(6,6),1)
        self.bn3 = nn.BatchNorm2d(128)
        self.tanh3 = nn.Tanh()
        self.drop3 = nn.Dropout(0.0)

        self.conv4 = nn.Conv2d(128,128,(3,3),1,1)
        self.bn4 = nn.BatchNorm2d(128)
        self.tanh4 = nn.Tanh()
        self.drop4 = nn.Dropout(0.0)   
        
        self.pool2 = nn.MaxPool2d((3,3),2,1)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.bn5 = nn.BatchNorm2d(256)
        self.tanh5 = nn.Tanh()
        self.drop5 = nn.Dropout(0.0)

        self.conv6 = nn.Conv2d(256,256,3,2,1)
        self.bn6 = nn.BatchNorm2d(256)
        self.tanh6 = nn.Tanh()
        self.drop6 = nn.Dropout(0.0)
        
        self.avg = nn.AvgPool2d((3,3),1)
        
        self.fc1 = nn.Linear(2304,128)
        self.tanh7 = nn.Tanh()
        self.drop_fc = nn.Dropout(0.0)
        self.fc2 = nn.Linear(128,6)
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,1)  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,1)  
    
    def forward(self,x):
        x = self.drop1(self.tanh1(self.bn1(self.conv1(x))))
        x = self.drop2(self.tanh2(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop3(self.tanh3(self.bn3(self.conv3(x))))
        x = self.drop4(self.tanh4(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.drop5(self.tanh5(self.bn5(self.conv5(x))))
        x = self.drop6(self.tanh6(self.bn6(self.conv6(x))))
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.drop_fc(self.tanh7(self.fc1(x)))
        x = self.fc2(x)

        return x  
    
    def update(self,d):
        self.drop_fc = nn.Dropout(d)
        return d
    
    def update_drop(self,i):
        drop_rate = math.log(i+3) /6
        
        if drop_rate < 0.9:
            self.drop_fc = nn.Dropout(drop_rate)
            return drop_rate
        else: 
            self.drop_fc = nn.Dropout(0.9)
            return 0.9

from torchsummary import summary

model = CNN_INTEL_RELU_drop_sched_0()
model.to('cuda')
summary(model, (3, 150, 150))