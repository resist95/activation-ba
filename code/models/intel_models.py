from sympy import Q
import torch
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


'''class CNN_INTEL(nn.Module):
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

        return x'''

'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),3)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2) 
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(156)
        self.fc1 = nn.Linear(2496,128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)

        return x'''

'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.conv5 = nn.Conv2d(156,215,(3,3),1,1) 
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(156)
        self.fc1 = nn.Linear(3440,128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.1)
        self.dropout6 = nn.Dropout(0.3)
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        #neu
        x = F.relu(self.conv5(x))
        x = self.dropout5(x)
        #neu
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout6(x)
        x = self.fc2(x)

        return x'''
#morgen: conv3 96-128 anstatt 96-156
#0.803 epoche 16
'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(156)
        self.fc1 = nn.Linear(2496,248)
        self.fc2 = nn.Linear(248,128)
        self.fc3 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.1)
        self.dropout6 = nn.Dropout(0.3)
        
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout6(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x'''

'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2496,128)
        self.fc2 = nn.Linear(128,6)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.3)
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = self.dropout5(x)
        x = self.fc2(x)

        return x'''
#fc2 entfernt
# 0.80         
'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2496,6)
     
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#size 64
# overfitting epoche 6
'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,64,5,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(64,64,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2,2),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(64,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2496,6)
     
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''

#dropout entfernt
# 3,3 pooling
# 0.8 epoche 10
# 5,5 pooling
# 0.82 epoche 8
'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((4,4),2)
        self.pool2 = nn.MaxPool2d((2,2),2)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1404,6)
     
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        #x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        #x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        #x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        #x = self.dropout4(x)

        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''

# pool2 nach conv4
# 
'''class CNN_INTEL(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,96,5,2)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((5,5),2)
        self.pool2 = nn.MaxPool2d((3,3),1)
        self.conv3 = nn.Conv2d(96,156,(5,5),2)
        self.conv4 = nn.Conv2d(156,156,(3,3),2,1)
        self.bn3 = nn.BatchNorm2d(156)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(156,6)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        #x = self.dropout2(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        #x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        #x = self.dropout4(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [12 / 100]
#Test acc: 0.75997150997151 Test loss: 0.7159378078452711 Train acc: 0.8470308788598575 Train loss: 0.413

'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)

        self.fc1 = nn.Linear(12800,6)
    
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#poch: [8 / 100]
#Test acc: 0.7350427350427351 Test loss: 0.7518741062267884 Train acc: 0.7960411718131433 Train loss: 0.55
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(3,3),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)

        self.fc1 = nn.Linear(15488,6)
    
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [10 / 100]
#Test acc: 0.7143874643874644 Test loss: 0.8043251429722894 Train acc: 0.7825019794140934 Train loss: 0.57
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,2)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)

        self.fc1 = nn.Linear(2048,6)
    
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [10 / 100]
#Test acc: 0.7321937321937322 Test loss: 0.7684923153276052 Train acc: 0.7588281868566904 Train loss: 0.661
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)

        self.fc1 = nn.Linear(12800,6)
    
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [9 / 100]
#Test acc: 0.7578347578347578 Test loss: 0.7048494618756872 Train acc: 0.7943784639746635 Train loss: 0.56542131
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)

        self.fc1 = nn.Linear(10368,6)
    
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [7 / 100]
#Test acc: 0.73005698005698 Test loss: 0.7330883352860635 Train acc: 0.7490894695170229 Train loss: 0.689909300
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)

        self.fc1 = nn.Linear(41472,6)
    
    #bn layer deaktivieren bei 128 batch size
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''

#Epoch: [13 / 100]
#Test acc: 0.7350427350427351 Test loss: 0.724207012520608 Train acc: 0.7929532858273951 Train loss: 0.5595894635821195
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)
        self.conv5 = nn.Conv2d(128,128,3,1)
        self.fc1 = nn.Linear(8192,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [12 / 100]
#Test acc: 0.7443019943019943 Test loss: 0.7636826104762745 Train acc: 0.8291369754552652 Train loss: 0.46794006379191927
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(3,3),2)
        self.conv5 = nn.Conv2d(128,256,3,1)
        self.fc1 = nn.Linear(16384,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [11 / 100]
#Test acc: 0.7471509971509972 Test loss: 0.7018046208294333 Train acc: 0.8119556611243072 Train loss: 0.5090520747501817
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)
        self.conv5 = nn.Conv2d(128,256,3,1)
        self.fc1 = nn.Linear(82944,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)
        self.conv5 = nn.Conv2d(128,256,5,1)
        self.fc1 = nn.Linear(65536,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [12 / 100]
#Test acc: 0.73005698005698 Test loss: 0.7449271903737757 Train acc: 0.7788598574821852 Train loss: 0.6024686116867534
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)
        self.conv5 = nn.Conv2d(128,256,3,2)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.conv7 = nn.Conv2d(256,512,3,1)
        self.fc1 = nn.Linear(12800,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [9 / 100]
#Test acc: 0.7806267806267806 Test loss: 0.6319954170979262 Train acc: 0.8110847189231988 Train loss: 0.5159533856170603
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)

        self.fc1 = nn.Linear(1152,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [10 / 100]
#Test acc: 0.7656695156695157 Test loss: 0.6490664320978147 Train acc: 0.8062549485352336 Train loss: 0.5349984594975248

#kernel (5,5),1 self.conv3
#Test acc: 0.7770655270655271 Test loss: 0.650671384813964 Train acc: 0.8057007125890736 Train loss: 0.5255824294673933
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)
        self.conv5 = nn.Conv2d(128,256,3,1)

        self.fc1 = nn.Linear(256,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [11 / 100]
#Test acc: 0.7863247863247863 Test loss: 0.5783532190704933 Train acc: 0.8133808392715756 Train loss: 0.5112386476351979
#Epoch: [10 / 100]
#Test acc: 0.7564102564102564 Test loss: 0.6827809525743541 Train acc: 0.7749802058590657 Train loss: 0.6203349846029448
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.conv3 = nn.Conv2d(96,128,(3,3),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)
        self.conv5 = nn.Conv2d(128,256,3,1)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.fc1 = nn.Linear(6400,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''

'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,2)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),2)
        self.conv5 = nn.Conv2d(128,256,3,1)
        
        self.fc1 = nn.Linear(36864,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Test acc: 0.7336182336182336 Test loss: 0.7813787896673543 Train acc: 0.7369754552652414 Train loss: 0.7065948581886656
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.pool2 = nn.MaxPool2d((3,3),1)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),1)
        self.conv5 = nn.Conv2d(128,256,3,1)
        
        self.fc1 = nn.Linear(30976,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [14 / 100]
#Test acc: 0.7528490028490028 Test loss: 0.7899453111246901 Train acc: 0.842596991290578 Train loss: 0.4352871007622428
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.pool2 = nn.MaxPool2d((3,3),2)
        self.conv3 = nn.Conv2d(96,128,(5,5),1)
        self.conv4 = nn.Conv2d(128,128,(5,5),1)
        self.conv5 = nn.Conv2d(128,256,3,1)
        
        self.fc1 = nn.Linear(2304,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#0.71
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((5,5),1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(5,5),1)
        self.conv5 = nn.Conv2d(128,256,3,2)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.conv7 = nn.Conv2d(256,512,3,1)
        self.conv8 = nn.Conv2d(512,1024,3,1)
        
        self.fc1 = nn.Linear(1024,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [15 / 100]
#Test acc: 0.7627360171001069 Test loss: 0.673574323636831 Train acc: 0.8288946290193284 Train loss: 0.46453221865381644
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((5,5),1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(5,5),1)
        self.conv5 = nn.Conv2d(128,256,3,2)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.conv7 = nn.Conv2d(256,512,3,1)
        self.conv8 = nn.Conv2d(512,512,3,1)
        
        self.fc1 = nn.Linear(512,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,2)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((5,5),1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(5,5),1)
        self.conv5 = nn.Conv2d(128,256,3,2)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.conv7 = nn.Conv2d(256,512,3,1)
        self.conv8 = nn.Conv2d(512,512,3,1)
        
        self.fc1 = nn.Linear(25088,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [10 / 100]
#Test acc: 0.7481296758104738 Test loss: 0.7189360430015236 Train acc: 0.7706422018348624 Train loss: 0.6285738034043438
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,3)
        self.conv2 = nn.Conv2d(96,96,3,1)
        self.pool1 = nn.MaxPool2d((5,5),1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2)
        self.conv4 = nn.Conv2d(128,128,(5,5),1)
        self.conv5 = nn.Conv2d(128,256,3,2)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.conv7 = nn.Conv2d(256,512,3,1)
        self.conv8 = nn.Conv2d(512,512,3,1)
        
        self.fc1 = nn.Linear(512,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [13 / 100]
#Test acc: 0.7452796579978624 Test loss: 0.7255786415138522 Train acc: 0.8123274249576913 Train loss: 0.5185695814357459
'''class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,96,5,4)
        self.conv2 = nn.Conv2d(96,96,3,1,1)
        self.pool1 = nn.MaxPool2d((5,5),1)
        self.conv3 = nn.Conv2d(96,128,(5,5),2,1)
        self.conv4 = nn.Conv2d(128,128,(5,5),1,1)
        self.conv5 = nn.Conv2d(128,256,3,2,1)
        self.conv6 = nn.Conv2d(256,256,3,1)
        self.conv7 = nn.Conv2d(256,512,3,1)
        self.conv8 = nn.Conv2d(512,512,3,1)
        
        self.fc1 = nn.Linear(512,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x'''
#Epoch: [17 / 100]
#Test acc: 0.7709298183113644 Test loss: 0.7119160963017122 Train acc: 0.8317449006858466 Train loss: 0.470395382375359
class CNN_INTEL(nn.Module):
    def __init__(self):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,32,5,3)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool1 = nn.MaxPool2d((3,3),2)
        self.conv3 = nn.Conv2d(64,64,5,1)
        self.conv4 = nn.Conv2d(64,128,5,1)
        self.conv5 = nn.Conv2d(128,128,3,1,1)
        self.conv6 = nn.Conv2d(128,128,5,1)
        self.conv7 = nn.Conv2d(128,24,3,1)
        self.fc1 = nn.Linear(24,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x
from torchsummary import summary
model = CNN_INTEL()
model.to('cuda')
summary(model, (3, 150, 150))