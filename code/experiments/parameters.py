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
#    'lr_relu' : 0.00003,
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

params_dict_cifar_algo = {
    #Parameter für normales dropout
    'lr_relu_normal': 0.00003,
    'lr_swish_normal': 0.00007,
    'lr_tanh_normal': 0.00001,
   
    #Parameter für log dropout
    #'lr_relu_log': ,
    #'lr_swish_log':,
    #'lr_tanh_log':,
    #'log_relu' : 2.05,
    #'log_swish' : 2.02,
    #'log_tanh' : 1.98,

    #Parameter für cur dropout
    'lr_relu_cur': 0.0003,
    'lr_swish_cur': 0.0003,
    'lr_tanh_cur': 0.0003,
    'cur_relu' : 0.0007,
    'cur_swish' : 0.0005,
    'cur_tanh' : 0.0002,

    #Parameter für ann dropout
    'lr_relu_ann': 0.00002,
    'lr_swish_ann': 0.00005,
    'lr_tanh_ann': 0.00005,
    'ann_relu' : 0.000008,
    'ann_swish' : 0.000001,
    'ann_tanh' : 0.000007,
}


params_dict_cifar = {
    'model' : 'CIFAR',
    'lr_relu' : 0.0001,
    'lr_swish': 0.0001,
    'lr_tanh' : 0.0001,
    'log_relu' : 2.01,
    'log_swish' : 2.01,
    'log_tanh' : 2.01,
    'cur_relu' : 0.0003,
    'cur_swish' : 0.0003,
    'cur_tanh' : 0.0003,
    'ann_relu' : 0.000008,
    'ann_swish' : 0.000001,
    'ann_tanh' : 0.000007,
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

#    'lr_relu_ann' : 0.000001,
#    'lr_swish_ann': 0.000009,
#    'lr_tanh_ann' : 0.000007,

params_dict_intel_algo = {
    #'lr_relu_normal': ,
    #'lr_swish_normal':,
    #'lr_tanh_normal':,
    #'lr_relu_log': ,
    #'lr_swish_log':,
    #'lr_tanh_log':,
    #'lr_relu_cur': ,
    #'lr_swish_cur':,
    #'lr_tanh_cur':,
    #'lr_relu_ann': ,
    #'lr_swish_ann':,
    #'lr_tanh_ann':,
    #'log_relu' : ,
    #'log_swish' : ,
    #'log_tanh' : ,
    #'cur_relu' : ,
    #'cur_swish' : ,
    #'cur_tanh' : ,
    #'ann_relu' : ,
    #'ann_swish' : ,
    #'ann_tanh' :,
}


params_dict_intel = {
    'model' : 'INTEL',
    'lr_relu' : 0.000008,
    'lr_swish': 0.000009,
    'lr_tanh' : 0.000007,
    'log_relu' : 2.00,
    'log_swish' : 2.00,
    'log_tanh' : 2.05,
    #'cur_relu' : ,
    #'cur_swish' : ,
    #'cur_tanh' : ,
    #'ann_relu' : ,
    #'ann_swish' : ,
    #'ann_tanh' : ,
    'batch_size' : 64,
    'classes' : 6,
    'max_epochs' :75,
    'weight_decay' : 0.000125
}

'''''lr_relu_drop_ann': 0.00007,
    'lr_swish_drop_ann':0.00007,
    'lr_tanh_ann':0.00007,
    '''
params_dict_mnist_algo = {

    #Parameter für normales dropout #x überprüft
    'lr_relu_normal': 0.0002,
    'lr_swish_normal': 0.0003,
    'lr_tanh_normal': 0.0002,

    #Parameter für log dropout #x überprüft
    'log_relu' : 1.9,
    'log_swish' : 1.9,
    'log_tanh' : 1.9,
    'lr_relu_drop_log': 0.00015,
    'lr_swish_drop_log': 0.00015,
    'lr_tanh_drop_log': 0.00009,

    #Parameter für curriculum dropout  #x überprüft
    'cur_relu' : 0.0007,
    'cur_swish' : 0.0002,
    'cur_tanh' : 0.0005,
    'lr_relu_drop_cur': 0.0001,
    'lr_swish_drop_cur': 0.0001,
    'lr_tanh_drop_cur': 0.0001,

    #Parameter für annealed dropout
    'ann_relu' : 0.00007,
    'ann_swish' : 0.00007,
    'ann_tanh' : 0.00007,
    'lr_relu_drop_ann': 0.0001,
    'lr_swish_drop_ann':0.0001,
    'lr_tanh_drop_ann':0.0001,
}

params_dict_mnist = {
    'model' : 'MNIST',
    'batch_size' : 128,
    'classes' : 10,
    'max_epochs' :100,
    'weight_decay': 0.000125
}