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