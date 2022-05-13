import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ray import tune
from ray.tune import CLIReporter
import os
import sys
#für gradienten lr 1e-5
params_dict_cifar = {
    'lr_relu' : 0.1,
    'lr_swish': 0.1,
    'lr_tanh' : 0.1,
    'batch_size' : 64,
    'max_epochs' :10,
    'classes' : 10,
    'weight_decay': 0.000125
}

params_dict_intel = {
    'lr_relu' : 0.0001,
    'lr_swish': 0.0001,
    'lr_tanh' : 0.0001,
    'batch_size' : 64,
    'classes' : 6,
    'max_epochs' :25,
    'weight_decay' : 0.000125
}

params_dict_mnist = {
    'lr_relu' : 0.0001,
    'lr_swish': 0.0001,
    'lr_tanh' : 0.0001,
    'batch_size' : 128,
    'classes' : 10,
    'max_epochs' :1,
    'weight_decay': 0.000125
}