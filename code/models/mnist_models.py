import torch.nn as nn
import torch.nn.functional as F


#params
# batch_size = 128
#n_epochs = 30-40
#lr = 0.001
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
        self.act_fn = act_fn
        self.act_fn_name = name
        self.layers = self._make_layers(self.act_fn)

        self.fc1 = nn.Linear(2704,128)
        self.fc2 = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.25)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self,act_fn):
        model = nn.Sequential(
            nn.Conv2d(1,8,3,1),
            nn.BatchNorm2d(8),
            act_fn(),
            
            nn.MaxPool2d(2,2),

            nn.Conv2d(8,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.25),
            act_fn(),
        )
        return model
    
    def forward(self,x):
        out = self.layers(x)
        out = out.reshape(out.shape[0],-1)
        if(self.act_fn_name == 'relu'):
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'tanh'):
            out = F.tanh(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'swish'):
            out = F.silu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'leakyrelu'):
            out = F.leaky_relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out
        elif(self.act_fn_name == 'gelu'):
            out = F.gelu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
            return out

from torchsummary import summary

#model = CNN_MNIST_ACT(nn.ReLU,'relu')
#summary(model,(1,28,28))