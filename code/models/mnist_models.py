import torch.nn as nn
import torch.nn.functional as F


#params
# batch_size = 128
#n_epochs = 15
#lr = 0.0001
class CNN_MNIST(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_MNIST,self).__init__()
        self.layers = self._make_layers()

        self.fc1 = nn.Linear(2704,128)
        self.fc2 = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.25)
    
    def _make_layers(self):
        model = nn.Sequential(
            nn.Conv2d(1,8,3,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.MaxPool2d(2,2),

            nn.Conv2d(8,16,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(16),
        )
        return model
    
    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CNN_MNIST_ACT(nn.Module):
    def __init__(self,act_fn,name):
        super(CNN_MNIST_ACT,self).__init__()
        self.act_fn = act_fn()
        self.act_fn_name = name

        self.conv1 = nn.Conv2d(1,8,3,1)
        self.conv2 = nn.Conv2d(8,16,3,1,1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(2704,128)
        self.fc2 = nn.Linear(128,10)
        self.soft = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        self.swish = nn.SiLU()
        self.gelu = nn.GELU()
        if self.act_fn_name == 'tanh':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d)):
                    nn.init.constant_(m.weight,1)
                    nn.init.constant_(m.bias, 0)
        else:            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        x = self.act_fn(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)

        x = self.act_fn(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0],-1)

        x = self.act_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from torchsummary import summary

#model = CNN_MNIST_ACT(nn.ReLU,'relu')
#summary(model,(1,28,28))