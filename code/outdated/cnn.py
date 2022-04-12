

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

########################################################
#CIFAR CNN FÜR WEITERE BEARBEITUNG
class CNN_CIFAR(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_CIFAR,self).__init__()

        self.layers = self._make_layers()
        self.avgpool = nn.AvgPool2d(2,2,0)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512,10)

    def _make_layers(self):
        model = nn.Sequential(
            nn.Conv2d(3,64,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2),2),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2),2),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2,2),2),

            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,2),2),

            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm2d(512),
        )
        return model

    def forward(self,x):
        out = self.layers(x)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0],-1)
        out = self.dropout(out)
        out = self.fc(out)
        return F.softmax(out,dim=1)

##########################################################
#MNIST CNN für weitere Bearbeitung


#########################################################

class CNN_INTEL(nn.Module):
    def __init__(self,num_classes=6):
        super(CNN_INTEL,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,1) 
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.conv4 = nn.Conv2d(64,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
        
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)
        
        self.pool = nn.MaxPool2d(2,2)

        self.linear1 = nn.Linear(4096,256)
        self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(128,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        x = self.bn3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.dropout2(x)
        x = self.bn4(x)
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.dropout2(x)
        x = self.bn5(x)
        x = self.pool(x)

        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.linear1(x))
        x = self.dropout3(x)

        x = F.relu(self.linear2(x))
        x = self.dropout3(x)

        x = self.linear3(x)

        return x


act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}


class block(nn.Module):
  expansion = 1
  def __init__(self, act_fn, in_channels,out_channels,subsample=False,stride=1):
    super(block,self).__init__()

    #downsample used to do 1x1 convolution
    self.downsample = None

    #create layers with stride 1 or 2 depending on subsample

    self.layers = self._make_layers(in_channels,out_channels,stride,act_fn)
    
    if subsample:
      self.downsample = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        act_fn(),
        #1x1 convolution
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2,bias=False),  
      )
    else:
      out_channels = in_channels

  def _make_layers(self,in_channels,out_channels,stride,act_fn):
    strides = int(stride)
    model = nn.Sequential(
      nn.BatchNorm2d(in_channels),
      act_fn(),
      nn.Conv2d(in_channels,out_channels,3,stride=strides,padding=1,bias=False),
      nn.BatchNorm2d(out_channels),
      act_fn(),
      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
    )
    return model
      
  def forward(self,x):
    net = self.layers(x)
    if self.downsample is None:
      return net
    else:
      down = self.downsample(x)
      net += down
      return net


class resnet(nn.Module):
  def __init__(self,block,in_channels,num_blocks,num_classes,act_fn):
    super(resnet,self).__init__()
    self.in_channels = 16
    self.act_fn_name = act_fn
    self.act_fn=act_fn_by_name[act_fn]
    self.conv1 = nn.Conv2d(in_channels,16,kernel_size=3,stride=1,padding=1,bias=False)
    
    self.layer1 = self._make_layers(block,
                  first_block=True,
                  act_fn=self.act_fn,
                  out_channels=16,
                  num_blocks=num_blocks[0],
                  stride=1)
    
    self.layer2 = self._make_layers(block,
                  first_block=False,
                  act_fn=self.act_fn,
                  out_channels=32,
                  num_blocks=num_blocks[1],
                  stride=2)
    
    self.layer3 = self._make_layers(block,
                  first_block=False,
                  act_fn=self.act_fn,
                  out_channels=64,
                  num_blocks=num_blocks[2],
                 stride=2)
    
    self.linear = nn.Linear(64,num_classes)

    self._weights_init()

  def _weights_init(self):
    #print(classname)
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity=self.act_fn_name)
      elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)  
    

  def _make_layers(self,block,first_block,act_fn,out_channels,num_blocks,stride):

    layers = []

    #Create array with ones but change stride of first entry depending on block
    strides = np.ones(num_blocks)
    strides[0] = stride

    if first_block:
      #Very first block no subsampling

      for s in strides:
        layers.append(block(act_fn,self.in_channels,out_channels,False,s))
        self.in_channels = out_channels *1
      return nn.Sequential(*layers)
    
    else:
      #subsample first block
      subsample=True
      for s in strides:
        layers.append(block(act_fn,self.in_channels,out_channels,subsample,s))
        subsample=False
        self.in_channels = out_channels
      return nn.Sequential(*layers)

  def forward(self,x):
    out = F.relu(self.conv1(x))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)

    out = F.avg_pool2d(out,out.size()[3])
    out = out.reshape(out.shape[0],-1)
    out = self.linear(out)
    return out


#function to access CNN's
def cnn_cifar(act_fn):
  return CNN_CIFAR()

def cnn_mnist(act_fn):
  return CNN_MNIST()

def cnn_intel(act_fn):
  return CNN_INTEL()    

#function to access resnets
def resnet20(size,classes,act_fn='relu'):
    return resnet(block,size,[3, 3, 3],classes,act_fn)


def resnet32(size,classes,act_fn='relu'):
    return resnet(block,size,[5, 5, 5],classes,act_fn)


def resnet44(size,classes,act_fn='relu'):
    return resnet(block,size,[7, 7, 7],classes,act_fn)


def resnet56(size,classes,act_fn='relu'):
    return resnet(block,size,[9, 9, 9],classes,act_fn)

#function to access special nets



def channel_shuffle(x, groups=2):
  bat_size, channels, w, h = x.shape
  group_c = channels // groups
  x = x.view(bat_size, groups, group_c, w, h)
  x = torch.transpose(x, 1, 2).contiguous()
  x = x.view(bat_size, -1, w, h)
  return x

# used in the block
def conv_1x1_bn(in_c, out_c, stride=1):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )

def conv_bn(in_c, out_c, stride=2):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )


class ShuffleBlock(nn.Module):
  def __init__(self, in_c, out_c, downsample=False):
    super(ShuffleBlock, self).__init__()
    self.downsample = downsample
    half_c = out_c // 2
    if downsample:
      self.branch1 = nn.Sequential(
          # 3*3 dw conv, stride = 2
          nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
          nn.BatchNorm2d(in_c),
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
      
      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 2
          nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
    else:
      # in_c = out_c
      assert in_c == out_c
        
      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 1
          nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
      
      
  def forward(self, x):
    out = None
    if self.downsample:
      # if it is downsampling, we don't need to do channel split
      out = torch.cat((self.branch1(x), self.branch2(x)), 1)
    else:
      # channel split
      channels = x.shape[1]
      c = channels // 2
      x1 = x[:, :c, :, :]
      x2 = x[:, c:, :, :]
      out = torch.cat((x1, self.branch2(x2)), 1)
    return channel_shuffle(out, 2)
    

class ShuffleNet2(nn.Module):
  def __init__(self, num_classes=10, input_size=224, net_type=1):
    super(ShuffleNet2, self).__init__()
    assert input_size % 32 == 0 
    
    
    self.stage_repeat_num = [4, 8, 4]
    if net_type == 0.5:
      self.out_channels = [3, 24, 48, 96, 192, 1024]
    elif net_type == 1:
      self.out_channels = [3, 24, 116, 232, 464, 1024]
    elif net_type == 1.5:
      self.out_channels = [3, 24, 176, 352, 704, 1024]
    elif net_type == 2:
      self.out_channels = [3, 24, 244, 488, 976, 2948]
    else:
      print("the type is error, you should choose 0.5, 1, 1.5 or 2")
      
    # let's start building layers
    self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    in_c = self.out_channels[1]
    
    self.stages = []
    for stage_idx in range(len(self.stage_repeat_num)):
      out_c = self.out_channels[2+stage_idx]
      repeat_num = self.stage_repeat_num[stage_idx]
      for i in range(repeat_num):
        if i == 0:
          self.stages.append(ShuffleBlock(in_c, out_c, downsample=True))
        else:
          self.stages.append(ShuffleBlock(in_c, in_c, downsample=False))
        in_c = out_c
    self.stages = nn.Sequential(*self.stages)
    
    in_c = self.out_channels[-2]
    out_c = self.out_channels[-1]
    self.conv5 = conv_1x1_bn(in_c, out_c, 1)
    self.g_avg_pool = nn.AvgPool2d(kernel_size=(int)(input_size/32)) # 如果输入的是224，则此处为7
    
    # fc layer
    self.fc = nn.Linear(out_c, num_classes)
    

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.stages(x)
    x = self.conv5(x)
    x = self.g_avg_pool(x)
    x = x.view(-1, self.out_channels[-1])
    x = self.fc(x)
    return x




class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class XceptionNet(nn.Module):
  def __init__(self,size,classes,act_fn='relu'):
    super(XceptionNet,self).__init__()
    self.act_fn=act_fn_by_name[act_fn]

    self.entry_flow = nn.Sequential(
      nn.Conv2d(size,32,kernel_size=3,stride=2,padding=1),
      nn.BatchNorm2d(32),
      self.act_fn(),

      nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(64),
      self.act_fn()
    )

