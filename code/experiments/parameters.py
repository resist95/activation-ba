import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import os
import sys

# fc drop und drop connect
#    'lr_relu' : 0.0005,
#    'lr_swish': 0.0007,
#    'lr_tanh' : 0.001,

# fc drop 0.8
#        'lr_relu' : 0.00003,
#    'lr_swish': 0.00007,
#    'lr_tanh' : 0.00001,

params_dict_cifar = {
    'lr_relu' : 0.00001,
    'lr_swish': 0.00001,
    'lr_tanh' : 0.00001,
    'batch_size' : 64,
    'max_epochs' :100,
    'classes' : 10,
    'weight_decay': 0.000125
}

#'    'lr_relu' : 0.00001,
#    'lr_swish': 0.00003,
#    'lr_tanh' : 0.00003,

params_dict_intel = {
    'lr_relu' : 0.00001,
    'lr_swish': 0.00001,
    'lr_tanh' : 0.00001,
    'batch_size' : 64,
    'classes' : 6,
    'max_epochs' :40,
    'weight_decay' : 0.000125
}
#    'lr_relu' : 0.0006,
#    'lr_swish': 0.0007,
#    'lr_tanh' : 0.0001,
params_dict_mnist = {
    'lr_relu' : 0.0001,
    'lr_swish': 0.0001,
    'lr_tanh' : 0.0001,
    'batch_size' : 128,
    'classes' : 10,
    'max_epochs' :50,
    'weight_decay': 0.000125
}