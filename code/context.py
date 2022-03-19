import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from neuralnetwork.preprocessing import ohc
from neuralnetwork.preprocessing import train_test_validate_split
from neuralnetwork.preprocessing import validate_split

from datasets import mnistdataset
from datasets import cifar10dataset
from datasets import caltech101dataset