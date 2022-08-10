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

params_dict_cifar_algo = {
    
    #[X] überprüft 1
    #[] logged
    #Parameter für normales dropout #x überprüft
    'lr_relu_normal': 0.00008,
    'lr_swish_normal': 0.0002,
    'lr_tanh_normal': 0.00004,
    
    #[X] überprüft 1
    #[X] überprüft 2
    #[] logged
    #Parameter für log dropout
    'lr_relu_drop_log': 0.0002, #0.0003 best 
    'lr_swish_drop_log': 0.00035, #0.00035 best
    'lr_tanh_drop_log': 0.00003, #0.00005 best 
    'log_relu' : 1.92, #1.92 best 82-83 2.00 fast of
    'log_swish' : 2.0, #2.00 best 84% 2.1 fast of
    'log_tanh' : 1.9,   #1.9 best 76% 1.98 fast of

    #[X] überprüft 1
    #[X] überprüft 2
    #[] logged
    #Parameter für cur dropout
    'lr_relu_drop_cur': 0.0002, #0.0002
    'lr_swish_drop_cur': 0.0006, #0.0006 best
    'lr_tanh_drop_cur': 0.00009, #0.00009
    'cur_relu' : 0.0009, #0.0009
    'cur_swish' : 0.0008, #0.0008
    'cur_tanh' : 0.0001, #0.0001

    #[] überprüft 1
    #[] überprüft 2
    #[] logged
    #Parameter für ann dropout
    'lr_relu_drop_ann': 0.00009, 
    'lr_swish_drop_ann': 0.0002, 
    'lr_tanh_drop_ann': 0.00003, 
    'ann_relu' : 0.00009 , #0.00009 best 
    'ann_swish' : 0.000008, #0.000008 best 
    'ann_tanh' : 0.000009, #0.00009 best 
}


params_dict_cifar = {
    'model' : 'CIFAR',
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

params_dict_intel_algo = {

    #[X] überprüft 1
    #[] logged
    #Parameter für dropout normal
    'lr_relu_normal': 0.0002, 
    'lr_swish_normal': 0.0002, #0.0002
    'lr_tanh_normal': 0.0002, #0.0002

    #[X] überprüft 1
    #[X] überprüft 2
    #[] logged
    #Parameter für dropout log
    'lr_relu_drop_log': 0.0002,
    'lr_swish_drop_log':0.0002,
    'lr_tanh_drop_log': 0.0002,
    'log_relu' : 2.00, 
    'log_swish' : 2.00,
    'log_tanh' : 2.00,

    #[] überprüft 1
    #[] überprüft 2
    #[] logged
    #Parameter für dropout cur
    'lr_relu_drop_cur' : 0.0002,
    'lr_swish_drop_cur': 0.0002,
    'lr_tanh_drop_cur' : 0.0002,
    'cur_relu' : 0.001,
    'cur_swish' : 0.001,
    'cur_tanh' : 0.001,

    #[] überprüft 1
    #[] überprüft 2
    #[] logged
    #Parameter für dropout ann
    'lr_relu_drop_ann' : 0.000001,
    'lr_swish_drop_ann': 0.000009,
    'lr_tanh_drop_ann' : 0.000007,
    'ann_relu' : 0.0001,
    'ann_swish' : 0.0001,
    'ann_tanh' : 0.0001,
}


params_dict_intel = {
    'model' : 'INTEL',
    'batch_size' : 64,
    'classes' : 6,
    'max_epochs' :75,
    'weight_decay' : 0.000125
}


params_dict_mnist_algo = {

    #[X] überprüft
    #[X] logged
    #Parameter für normales dropout #x überprüft
    'lr_relu_normal': 0.0002,
    'lr_swish_normal': 0.0003,
    'lr_tanh_normal': 0.0002,

    #[X] überprüft 1
    #[X] überprüft 2
    #[X] logged
    #Parameter für log dropout 
    'log_relu' : 1.9,
    'log_swish' : 1.9,
    'log_tanh' : 1.9,
    'lr_relu_drop_log': 0.00015,
    'lr_swish_drop_log': 0.00015,
    'lr_tanh_drop_log': 0.00009,

    #[X] überprüft 1
    #[X] überprüft 2
    #[X] logged
    #Parameter für curriculum dropout 
    'cur_relu' : 0.0007,
    'cur_swish' : 0.0002,
    'cur_tanh' : 0.0006,
    'lr_relu_drop_cur': 0.00009, 
    'lr_swish_drop_cur': 0.00009, 
    'lr_tanh_drop_cur': 0.00009,

    #[X] überprüft 1
    #[X] überprüft 2
    #[] logged
    #Parameter für annealed dropout 
    'ann_relu' : 0.00009,
    'ann_swish' : 0.00009,
    'ann_tanh' : 0.00009,
    'lr_relu_drop_ann': 0.00009, 
    'lr_swish_drop_ann':0.00009,
    'lr_tanh_drop_ann':0.00009,
}

params_dict_mnist = {
    'model' : 'MNIST',
    'batch_size' : 128,
    'classes' : 10,
    'max_epochs' :100,
    'weight_decay': 0.000125
}