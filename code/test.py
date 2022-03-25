import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchsummary import summary
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 15
num_classes = 10
batch_size = 100
learning_rate = 0.007

#custom datasets
class Intel(Dataset):
    
    def __init__(self,train=False):
        #Init des Dateipfads sowie laden der Daten
        self.path_to_dir = 'files/intel'
        self.X_train,self.y_train,self.X_test,self.y_test = self.__load()
        self.train = train

    def __load(self):
        # Laden der Daten mithilfe von glob das alle
        # Daten mit .jpg auflistet und liste hinzufügt
        data = []
        labels = []
        filename_train_data = self.path_to_dir + '/train/**/*.jpg'
        train_imgfiles = []
        for file in glob.glob(filename_train_data):
            train_imgfiles.append(file)

        train_data = []
        train_labels = []
        size = (150,150)
        for paths in train_imgfiles:
        # Labels an Liste hinzufügen
        # 
        # Zum Schluss umwandeln liste in ndarray
        # np.shape = ()
            train_labels.append(paths.split(os.path.sep)[-2])
            I = cv.imread(paths)
            rgb = cv.cvtColor(I,cv.COLOR_BGR2RGB)
            resized = cv.resize(rgb,size)
            train_data.append(resized)

        # Laden der Daten mithilfe von glob das alle
        # Daten mit .jpg auflistet und liste hinzufügt
        filename_test_data = self.path_to_dir + '/test/**/*.jpg'
        test_imgfiles = []
        for file in glob.glob(filename_test_data):
            test_imgfiles.append(file)

        test_data = []
        test_labels = []
        size = (150,150)
        for paths in test_imgfiles:
        # Labels an Liste hinzufügen
        # 
        # Zum Schluss umwandeln liste in ndarray
        # np.shape = ()
            test_labels.append(paths.split(os.path.sep)[-2])
            I = cv.imread(paths)
            rgb = cv.cvtColor(I,cv.COLOR_BGR2RGB)
            resized = cv.resize(rgb,size)
            test_data.append(resized)

        return train_data,train_labels,test_data,test_labels

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)
    
    def __getitem__(self,idx):
        
        if self.train:
            lb = LabelEncoder()
            lab = lb.fit_transform(self.y_train)
            convert_tensor = transforms.ToTensor()
            d = []
            d = convert_tensor(self.X_train[idx])
            return d,lab[idx]
        else:
            lb = LabelEncoder()
            lab = lb.fit_transform(self.y_test)
            convert_tensor = transforms.ToTensor()
            d = []
            d = convert_tensor(self.X_test[idx])
            return d,lab[idx]

class caltech101():

    def __init__(self,train=False):
        #Init des Dateipfads sowie laden der Daten
        self.path_to_dir = 'files/caltech101/101_ObjectCategories'
        data,labels = self.__load()
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(data,labels,test_size=0.2,train_size=0.8)
        self.train=train

    def __load(self):
        # Laden der Daten mithilfe von glob das alle
        # Daten mit .jpg auflistet und liste hinzufügt
        path_to_img = self.path_to_dir + '/*/*.jpg'
        imgfiles = []
        for file in glob.glob(path_to_img):
            imgfiles.append(file)

        data = []
        labels = []
        for paths in imgfiles:
        # Labels an Liste hinzufügen
        # Bilder resizen das alle gleiche groesse und dim haben
        # Zum Schluss umwandeln liste in ndarray
        # np.shape = (8676, 200, 200, 3)    
            labels.append(paths.split(os.path.sep)[-2])
            I = cv.imread(paths)
            data.append(I)
        return data,labels

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test) 

    def __getitem__(self,idx):
        if self.train:
            lb = LabelEncoder()
            lab = lb.fit_transform(self.y_train)
            convert_tensor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
            d = []
            d = convert_tensor(self.X_train[idx])
            return d,lab[idx]
        else:
            lb = LabelEncoder()
            lab = lb.fit_transform(self.y_test)
            convert_tensor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
            d = []
            d = convert_tensor(self.X_test[idx])
            return d,lab[idx]

# Load datasets
train_mnist = torchvision.datasets.MNIST(root='E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/mnist',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=False)

test_mnist = torchvision.datasets.MNIST(root='E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/mnist',
                                          train=False, 
                                          transform=transforms.ToTensor())

train_cifar = torchvision.datasets.CIFAR10(root='E:/Explorer/Dokumente/Bachelor/git/activation-ba/files',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=False)

test_cifar = torchvision.datasets.CIFAR10(root='E:/Explorer/Dokumente/Bachelor/git/activation-ba/files',
                                            train=False,
                                            transform=transforms.ToTensor())

train_caltech = caltech101(train=True)

test_caltech = caltech101(train=False)

train_intel = Intel(train=True)

test_intel = Intel(train=False)


# Data loader
train_loader_mnist = torch.utils.data.DataLoader(dataset=train_mnist,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader_mnist = torch.utils.data.DataLoader(dataset=test_mnist,
                                          batch_size=batch_size, 
                                          shuffle=False)

train_loader_cifar = DataLoader(dataset=train_cifar,
                                        batch_size=batch_size,
                                        shuffle=True)

test_loader_cifar = DataLoader(dataset=test_cifar,
                                        batch_size=batch_size,
                                        shuffle=False)

train_loader_caltech101 = DataLoader(dataset=train_caltech,
                                        batch_size=batch_size,
                                        shuffle=True)

test_loader_caltech101 = DataLoader(dataset=test_caltech,
                                        batch_size=batch_size,
                                        shuffle=False)

train_loader_intel = DataLoader(train_intel,
                                batch_size=batch_size,
                                shuffle=True)

train_loader_intel = DataLoader(test_intel,
                                batch_size=batch_size,
                                shuffle=True)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024,10)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out

class ConvNet_CIFAR(nn.Module):
    def __init__(self):
        super(ConvNet_CIFAR,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(8,4,kernel_size=3,stride=2,padding=1)

        self.dropout = nn.Dropout2d(0.1)

        self.dense1 = nn.Linear(1024,128)
        self.dense = nn.Softmax()
        = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU()
        )

    
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = self.conv3(x)
        out.reshape(out.size(0), -1)
        out = self.dense(x)
        return out

model = ConvNet(num_classes).to(device)
summary(model,(3,32,32))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader_cifar)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader_cifar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader_cifar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
