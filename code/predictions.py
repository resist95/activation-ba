from context import cifar10dataset
from context import caltech101dataset
from context import mnistdataset


train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1
cif = cifar10dataset(val_ratio)
mni = mnistdataset(val_ratio)
cal = caltech101dataset(train_ratio,test_ratio,val_ratio)

data = [cif,mni,cal]
dataset_names = ['CIFAR10','MNIST','CALTECH101']
