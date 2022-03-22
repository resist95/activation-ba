from context import cifar10dataset
from context import caltech101dataset
from context import mnistdataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    
    def __init__(self,images,labels,transform = None):
        self.data = images,
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx): 
        image = self.data[idx]
        data = {'image':image,'label':self.labels}
        if self.transform:
            data = self.transform(data)
        return data


train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1
cif = cifar10dataset(val_ratio)
mni = mnistdataset(val_ratio)
cal = caltech101dataset(train_ratio,test_ratio,val_ratio)

data = [cif,mni,cal]
dataset_names = ['CIFAR10','MNIST','CALTECH101']

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
]) 

cif_nn =   "#todo cnn"
mnist_nn = "#todo cnn"
cal_nn = "#todo cnn"

cnn = [cif_nn,mnist_nn,cal_nn]

for ds in data:
    dic = ds.get_split_data()
    train_data = CustomDataset(dic['X_train'],dic['y_train'],train_transforms)
    test_data = CustomDataset(dic['X_test'],dic['y_test'],val_transform)
    val_data = CustomDataset(dic['X_val'],dic['y_val'],val_transform)

    trainLoader = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=4)
    testLoader = DataLoader(test_data,batch_size=32,shuffle=True,num_workers=4)
    valLoader = DataLoader(val_data,batch_size=32,shuffle=True,num_workers=4)

