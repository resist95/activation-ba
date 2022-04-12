import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))


from datasets.datasets import Cifar10Dataset, IntelDataset, MnistDataset
from datasets.datasets_random_mean import IntelDataset_random_mean
from datasets.data import CIFAR10, Intel, MNIST
from models.cifar_models import CNN_CIFAR
from models.mnist_models import CNN_MNIST
from models.intel_models import CNN_INTEL
