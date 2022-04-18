import torch.nn as nn
import torch.nn.functional as F


class CNN_CIFAR(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_CIFAR,self).__init__()

        self.layers = self._make_layers()
        self.fc = nn.Linear(86528,10)
        self.dropout = nn.Dropout(0.4)

    def _make_layers(self):
        model = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.Conv2d(32,64,3),
            nn.Conv2d(64,128,3),
        )
        return model

    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        out = self.dropout(out)
        out = self.fc(out)
        return F.softmax(out,dim=1)



class CNN_CIFAR_2(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_CIFAR_2,self).__init__()

        self.layers = self._make_layers()
        self.fc = nn.Linear(768,10)
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.MaxPool2d(2,2)

    def _make_layers(self):
        model = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.Conv2d(32,64,3),
            nn.Conv2d(64,128,3),
        )
        return model

    def forward(self,x):
        out = self.layers(x)
        out = self.pool(x)
        out = out.reshape(out.shape[0],-1)
        out = self.dropout(out)
        out = self.fc(out)
        return F.softmax(out,dim=1)


class CNN_CIFAR_3(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_CIFAR_3,self).__init__()

        self.layers = self._make_layers()
        self.fc = nn.Linear(768,10)
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.MaxPool2d(2,2)

    def _make_layers(self):
        model = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
        )
        return model

    def forward(self,x):
        out = self.layers(x)
        out = self.pool(x)
        out = self.dropout(out)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return F.softmax(out,dim=1)


class CNN_CIFAR_4(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_CIFAR_3,self).__init__()

        self.layers = self._make_layers()
        self.fc = nn.Linear(768,10)
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.MaxPool2d(2,2)

    def _make_layers(self):
        model = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
        )
        return model

    def forward(self,x):
        out = self.layers(x)
        out = self.pool(x)
        out = self.dropout(out)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return F.softmax(out,dim=1)


class CNN_CIFAR_ACT(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_CIFAR_ACT,self).__init__()
        self.act_fn = act_fn()
        self.act_fn_name = name

        self.conv1 = nn.Conv2d(3,32,(3,3),1,1)
        self.conv2 = nn.Conv2d(32,32,(3,3),1,1)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.conv4 = nn.Conv2d(64,64,3,1,1)
        self.conv5 = nn.Conv2d(64,128,3,1,1)
        self.conv6 = nn.Conv2d(128,128,3,1,1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d((2,2),2)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool2d(2,2,0)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512,10)

    def forward(self,x):
        x = self.act_fn(self.conv1(x))
        #x = self.bn1(x)
        x = self.act_fn(self.conv2(x))
        #x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.act_fn(self.conv3(x))
        #x = self.bn2(x)
        x = self.dropout2(x)
        x = self.act_fn(self.conv4(x))
        #x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.act_fn(self.conv5(x))
        #x = self.bn3(x)
        x = self.act_fn(self.conv6(x))
        #x = self.bn3(x)
        x = self.pool(x)
        x = self.dropout4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x


'''class CNN_CIFAR_COLAB(nn.Module):
    def __init__(self):
        super(CNN_CIFAR_COLAB,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        #fc layers
        self.fc1 = nn.Linear(57600,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [20 / 100]
#Test acc: 0.5978 Test loss: 1.8087009490008634 Train acc: 0.9412444444444444 Train loss: 0.20457342743013437
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,32,3,1)
        #fc layers
        self.fc1 = nn.Linear(28800,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
# Epoch: [20 / 100]
#Test acc: 0.5966 Test loss: 2.027041741375039 Train acc: 0.9628444444444444 Train loss: 0.1295489573975615
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        #fc layers
        self.fc1 = nn.Linear(57600,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [20 / 100]
#Test acc: 0.5996 Test loss: 2.301798297010646 Train acc: 0.9753111111111111 Train loss: 0.08686006144410478
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(115200,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,128,5,1)
        #fc layers
        self.fc1 = nn.Linear(100352,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [20 / 100]
#Test acc: 0.5894 Test loss: 2.308555100284111 Train acc: 0.9675111111111111 Train loss: 0.11093061502902234
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,5,1)
        #fc layers
        self.fc1 = nn.Linear(50176,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [20 / 100]
#Test acc: 0.6116 Test loss: 1.4595201267015045 Train acc: 0.8750222222222223 Train loss: 0.38391314678672805
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        #fc layers
        self.fc1 = nn.Linear(14400,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [20 / 100]
#Test acc: 0.621 Test loss: 2.251998991138863 Train acc: 0.9556444444444444 Train loss: 0.13460701040110157
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        #fc layers
        self.fc1 = nn.Linear(10816,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [20 / 100] overfit ep 4
#Test acc: 0.6602 Test loss: 2.5307368036988978 Train acc: 0.9865333333333334 Train loss: 0.04129135051042412
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(21632,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit ep 8
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        #fc layers
        self.fc1 = nn.Linear(7744,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit ep 4
#Epoch: [10 / 100]
#Test acc: 0.6962 Test loss: 1.015825646101221 Train acc: 0.8602666666666666 Train loss: 0.4014197835139704
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,5,1)
        #fc layers
        self.fc1 = nn.Linear(5184,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit ep 6
#Epoch: [10 / 100]
#Test acc: 0.7182 Test loss: 0.9736283893366031 Train acc: 0.8832666666666666 Train loss: 0.33114026652487116
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,5,1)
        self.conv4 = nn.Conv2d(64,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(6272,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit 7

'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(10368,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit ep 8
#Epoch: [10 / 100]
#Test acc: 0.7076 Test loss: 1.1151543033464002 Train acc: 0.9036666666666666 Train loss: 0.275487160417368
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(10368,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit ep 9
#Epoch: [10 / 100]
#Test acc: 0.724 Test loss: 0.8673782237661256 Train acc: 0.8246 Train loss: 0.49390004329542636

#lr 0.0003
#Epoch: [14 / 100]
#Test acc: 0.7242 Test loss: 0.8004839751885765 Train acc: 0.7838888888888889 Train loss: 0.6118757399363317
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,5,1)
        #fc layers
        self.fc1 = nn.Linear(1152,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#overfit ep 9
#Epoch: [9 / 100]
#Test acc: 0.7082 Test loss: 0.8644120720461805 Train acc: 0.7941333333333334 Train loss: 0.5755213508797579
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,5,1)
        self.conv7 = nn.Conv2d(128,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(128,10)
    
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
#overfit ep 8
#Epoch: [8 / 100]
#Test acc: 0.7346 Test loss: 0.7836646074528029 Train acc: 0.7846666666666666 Train loss: 0.6115151942058482
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,5,1)
        self.conv7 = nn.Conv2d(128,256,3,1)
        #fc layers
        self.fc1 = nn.Linear(256,10)
    
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
#overfit ep 10
#Epoch: [10 / 100]
#Test acc: 0.7084 Test loss: 0.8799851545562849 Train acc: 0.8056666666666666 Train loss: 0.5464274031734282
#lr 0.0003
#overfit ep 19
#Epoch: [19 / 100]
#Test acc: 0.7134 Test loss: 0.8658144327143332 Train acc: 0.8139777777777778 Train loss: 0.5261223589636795
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(1152,10)
    
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
#Epoch: [15 / 100]
#Test acc: 0.7044 Test loss: 0.8697665031851936 Train acc: 0.7848888888888889 Train loss: 0.6079114065737035
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,1)
        self.conv8 = nn.Conv2d(128,256,3,1)
        #fc layers
        self.fc1 = nn.Linear(256,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#Epoch: [14 / 100]
#Test acc: 0.707 Test loss: 0.8420638849498177 Train acc: 0.7812444444444444 Train loss: 0.6199060773590033
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(1152,512)
        self.fc2 = nn.Linear(512,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = x.reshape(x.shape[0],-1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)

      return x'''
#overfit ep 16
#Epoch: [16 / 100]
#Test acc: 0.7066 Test loss: 0.8556214763475531 Train acc: 0.7934 Train loss: 0.5858255690085696
#overfit ep 8 lr 0.001
#Epoch: [8 / 100]
#Test acc: 0.6768 Test loss: 0.9483548700622059 Train acc: 0.7686888888888889 Train loss: 0.6558412146029
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,2)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,1)
        #fc layers
        self.fc1 = nn.Linear(1152,125)
        self.fc2 = nn.Linear(125,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = x.reshape(x.shape[0],-1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)

      return x'''
#overfit ep 9
#Epoch: [9 / 100]
#Test acc: 0.7184 Test loss: 0.818252319008557 Train acc: 0.7916666666666666 Train loss: 0.5879848622458006

#overfit ep 10
#Epoch: [10 / 100]
#Test acc: 0.7084 Test loss: 0.8799851545562849 Train acc: 0.8056666666666666 Train loss: 0.5464274031734282
#lr 0.0003
#overfit ep 19
#Epoch: [19 / 100]
#Test acc: 0.7134 Test loss: 0.8658144327143332 Train acc: 0.8139777777777778 Train loss: 0.5261223589636795
#score to beat
#lr 0.001
#Epoch: [7 / 100]
#Test acc: 0.7256 Test loss: 0.7877485503884145 Train acc: 0.7708444444444444 Train loss: 0.646130442959803
#lr 0.0003
#Epoch: [21 / 100]
#Test acc: 0.742 Test loss: 0.7835617146177354 Train acc: 0.8495111111111111 Train loss: 0.4295901087125086

#lr 0.001
#Epoch: [10 / 100]
#Test acc: 0.739 Test loss: 0.7887793023340428 Train acc: 0.8476666666666667 Train loss: 0.43542668256253736
#Epoch: [18 / 100]
#Test acc: 0.7538 Test loss: 0.7899098328062922 Train acc: 0.8482666666666666 Train 
#loss: 0.4244921815462876

#Epoch: [9 / 100]
#Test acc: 0.7438 Test loss: 0.7650421162830261 Train acc: 0.8284 Train loss: 0.4896141971044553
#Epoch: [15 / 100]
#Test acc: 0.7432 Test loss: 0.7777716549163872 Train acc: 0.8029555555555555 Train 
#loss: 0.5601002212550171

#Epoch: [8 / 100]
#Test acc: 0.7302 Test loss: 0.799756853447317 Train acc: 0.7910444444444444 Train loss: 0.5893240846618384
#Epoch: [15 / 100]
#Test acc: 0.7526 Test loss: 0.7657443207972416 Train acc: 0.8364 Train loss: 0.4649642377246835

#Epoch: [7 / 100]
#Test acc: 0.7188 Test loss: 0.8070010442728458 Train acc: 0.7674222222222222 Train 
#loss: 0.6566118032833026
#Epoch: [12 / 100]
#Test acc: 0.7156 Test loss: 0.8454773256422603 Train acc: 0.7966 Train loss: 0.5814204415305493

#Epoch: [13 / 100]
#Test acc: 0.7148 Test loss: 0.8243998209930756 Train acc: 0.8009555555555555 Train loss: 0.5682306512895533
#Epoch: [10 / 100]
#Test acc: 0.7116 Test loss: 0.8343734453732841 Train acc: 0.7910222222222222 Train loss: 0.5969069841864978
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)

        #fc layers
        self.fc1 = nn.Linear(10368,10)
    
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
#best stride 2 at 3
#Epoch: [11 / 100]
#Test acc: 0.7452 Test loss: 0.8089868168793611 Train acc: 0.8790444444444444 Train loss: 0.3390878917946018
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,2)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,1)

        #fc layers
        self.fc1 = nn.Linear(128,10)
    
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
#Epoch: [15 / 100]
#Test acc: 0.7066 Test loss: 0.8600255755882684 Train acc: 0.7857777777777778 Train loss: 0.6094839501770999
#letzte layer stride 2 noch testen
#Epoch: [20 / 100]
#Test acc: 0.76 Test loss: 0.7249853863427663 Train acc: 0.841555555555555


#Epoch: [12 / 100]
#Test acc: 0.7636 Test loss: 0.8094791477293612 Train acc: 0.8748222222222222 Train loss: 0.34891280742848463
#Epoch: [18 / 100]
#Test acc: 0.7432 Test loss: 0.8114963929261314 Train acc: 0.8380888888888889 Train loss: 0.4530005770898891
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)
        self.conv8 = nn.Conv2d(128,256,3,1)

        #fc layers
        self.fc1 = nn.Linear(256,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
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
#Test acc: 0.7292 Test loss: 0.8455932323458346 Train acc: 0.8369777777777778 Train loss: 0.4573922173096568
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)
        self.conv8 = nn.Conv2d(128,512,3,1)

        #fc layers
        self.fc1 = nn.Linear(512,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)

      return x'''

# 0.768
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)
        self.conv8 = nn.Conv2d(128,256,3,1)

        #fc layers
        self.fc1 = nn.Linear(1024,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
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
#Test acc: 0.75 Test loss: 0.8202833789865593 Train acc: 0.8613111111111111 Train loss: 0.3857734756127466
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)
        self.conv8 = nn.Conv2d(128,256,3,1,1)
        self.conv9 = nn.Conv2d(256,256,3,1,1)

        #fc layers
        self.fc1 = nn.Linear(4096,10)
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = F.relu(self.conv9(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)

      return x'''
#lr 0.0003
#Epoch: [18 / 100]
#Test acc: 0.7424 Test loss: 0.7751532425325457 Train acc: 0.8294888888888889 Train loss: 0.4884323997690024
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,128,3,1)
        self.conv2 = nn.Conv2d(128,128,3,1)
        self.conv3 = nn.Conv2d(128,128,3,2)
        self.conv4 = nn.Conv2d(128,256,3,1)

        self.conv5 = nn.Conv2d(256,256,3,1)
        self.conv6 = nn.Conv2d(256,256,3,1)
        #fc layers
        self.fc1 = nn.Linear(12544,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,128,3,1)
        self.conv2 = nn.Conv2d(128,128,3,1)
        self.conv3 = nn.Conv2d(128,128,3,2)
        self.conv4 = nn.Conv2d(128,256,3,1)

        self.conv5 = nn.Conv2d(256,256,3,1)
        self.conv6 = nn.Conv2d(256,256,5,1)
        #fc layers
        self.fc1 = nn.Linear(6400,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#dauert zu lang
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,128,3,1)
        self.conv2 = nn.Conv2d(128,128,3,1)
        self.conv3 = nn.Conv2d(128,128,3,2)
        self.conv4 = nn.Conv2d(128,256,3,1)

        self.conv5 = nn.Conv2d(256,256,3,1)
        self.conv6 = nn.Conv2d(256,512,3,1)
        #fc layers
        self.fc1 = nn.Linear(25088,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)
      return x'''
#top performer mit bs 64
#Epoch: [13 / 100]
#Test acc: 0.7548 Test loss: 0.7439062518061502 Train acc: 0.8233555555555555 Train loss: 0.4967381403617363
#bs 32
#Epoch: [10 / 100]
#Test acc: 0.7508 Test loss: 0.7639300803591079 Train acc: 0.8397333333333333 Train loss: 0.4538904773511822
#bs 256
#Epoch: [12 / 100]
#Test acc: 0.7526 Test loss: 0.7560362504520767 Train acc: 0.8327777777777777 Train loss: 0.4686362056141999
'''class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)
        self.conv8 = nn.Conv2d(128,256,3,1)

        #fc layers
        self.fc1 = nn.Linear(256,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)

      return x'''
class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR,self).__init__()
        #conv layers
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.pool1 = nn.MaxPool2d((2,2),1)
        self.pool2 = nn.MaxPool2d((2,2),1)
        self.pool3 = nn.MaxPool2d((2,2),1)
        self.conv2 = nn.Conv2d(64,64,3,1)
        self.conv3 = nn.Conv2d(64,64,3,2)
        self.conv4 = nn.Conv2d(64,128,3,1,1)

        self.conv5 = nn.Conv2d(128,128,3,1)
        self.conv6 = nn.Conv2d(128,128,3,1)
        self.conv7 = nn.Conv2d(128,128,3,2)
        self.conv8 = nn.Conv2d(128,256,3,1)

        #fc layers
        self.fc1 = nn.Linear(256,10)
    
    def forward(self,x):
      x = F.relu(self.conv1(x))
      x = F.relu(self.conv2(x))
      x = self.pool1(x)
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      x = self.pool2(x)
      x = F.relu(self.conv5(x))
      x = F.relu(self.conv6(x))
      x = self.pool3(x)
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      x = x.reshape(x.shape[0],-1)
      x = self.fc1(x)

      return x
from torchsummary import summary
model = CNN_CIFAR()
model.to('cuda')
summary(model, (3, 32, 32))