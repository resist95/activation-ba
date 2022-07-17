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


#'    'lr_relu' : 0.00003
#    'lr_swish': 0.00007
#    'lr_tanh' : 0.00001

#    'lr_relu_log' : ,
#    'lr_swish_log': ,
#    'lr_tanh_log' : ,

#    'lr_relu_cur' : 0.0001,
#    'lr_swish_cur': 0.0001,
#    'lr_tanh_cur' : 0.0001,

#    'lr_relu_ann' : 0.00001,
#    'lr_swish_ann': 0.00001,
#    'lr_tanh_ann' : 0.00001,

params_dict_cifar = {
    'model' : 'CIFAR',
    'lr_relu' : 0.00002,
    'lr_swish': 0.00003,
    'lr_tanh' : 0.00001,
    'log_relu' : 2.00,
    'log_swish' : 2.00,
    'log_tanh' : 2.05,
    'batch_size' : 64,
    'max_epochs' :75,
    'classes' : 10,
    'weight_decay': 0.000125
}

#'    'lr_relu' : 0.000005,
#    'lr_swish': 0.000007,
#    'lr_tanh' : 0.000005,

#    'lr_relu_log' : 0.000005,
#    'lr_swish_log': 0.000007,
#    'lr_tanh_log' : 0.000005,

#    'lr_relu_cur' : 0.000005,
#    'lr_swish_cur': 0.000007,
#    'lr_tanh_cur' : 0.000005,

#    'lr_relu_ann' : 0.00001,
#    'lr_swish_ann': ,
#    'lr_tanh_ann' : ,
params_dict_intel = {
    'model' : 'INTEL',
    'lr_relu' : 0.00001,
    'lr_swish': 0.00001,
    'lr_tanh' : 0.00001,
    'log_relu' : 2.00,
    'log_swish' : 2.00,
    'log_tanh' : 2.05,
    'batch_size' : 64,
    'classes' : 6,
    'max_epochs' :75,
    'weight_decay' : 0.000125
}
#    'lr_relu' : 0.0006,
#    'lr_swish': 0.0007,
#    'lr_tanh' : 0.0001,

#    'lr_relu_log' : ,
#    'lr_swish_log':,
#    'lr_tanh_log' : ,

#    'lr_relu_cur' : ,
#    'lr_swish_cur': ,
#    'lr_tanh_cur' : ,

#    'lr_relu_ann' : ,
#    'lr_swish_ann': ,
#    'lr_tanh_ann' : ,
params_dict_mnist = {
    'model' : 'MNIST',
    'lr_relu' : 0.0001,
    'lr_swish': 0.0003,
    'lr_tanh' : 0.0003,
    'log_relu' : 2.05,
    'log_swish' : 2.03,
    'log_tanh' : 2.02,
    'batch_size' : 128,
    'classes' : 10,
    'max_epochs' :100,
    'weight_decay': 0.000125
}