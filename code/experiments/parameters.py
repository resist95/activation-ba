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
    'lr_relu' : 0.001,
    'lr_swish': 0.0007,
    'lr_tanh' : 0.0003,
    'batch_size' : 64,
    'max_epochs' :30,
    'weight_decay': 0.000125
}

params_dict_intel = {
    'lr_relu' : 0,
    'lr_swish': 0,
    'lr_tanh' : 0,
    'batch_size' : 0,
    'max_epochs' :0
}

params_dict_mnist = {
    'lr_relu' : 0.0005,
    'lr_swish': 0.0005,
    'lr_tanh' : 0.0005,
    'batch_size' : 128,
    'max_epochs' :15,
    'weight_decay': 0.000125
}