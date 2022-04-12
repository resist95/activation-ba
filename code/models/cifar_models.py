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
        x = self.bn1(x)
        x = self.act_fn(self.conv2(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.act_fn(self.conv3(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.act_fn(self.conv4(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.act_fn(self.conv5(x))
        x = self.bn3(x)
        x = self.act_fn(self.conv6(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.dropout4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x


from torchsummary import summary
model = CNN_CIFAR_ACT(nn.ReLU,'relu')
summary(model, (3, 32, 32))