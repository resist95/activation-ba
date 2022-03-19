import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchvision import  transforms

from torch.utils.data import Dataset

def train_test_validate_split(train_ratio, test_ratio, validation_ratio, X, y):
    #X = Daten
    #y = Label
    #Teilt Daten in train,test,validate
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1-train_ratio,shuffle=True)
    X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=test_ratio / (test_ratio + validation_ratio))
    return X_train,X_test,X_val,y_train,y_test,y_val

def validate_split(test_ratio, validation_ratio, X, y):
    X_val,X_test,y_val,y_test = train_test_split(X,y,test_size=test_ratio / (test_ratio + validation_ratio))
    return X_val,y_val

def transform(mean,std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean),(std))
    ])
    return transform

def ohc(X,y,already_split=True):
    enc = OneHotEncoder()
    enc.fit(y)
    if already_split:
        X_val,y_val = \
            validate_split(0.2,0.1,X,y)
        X_val = enc.transform(X_val).toarray()
        dic = {'X_val': X_val, 
            'y_val': y_val}
    else:
        X_train,X_test,X_val,y_train,y_test,y_val = \
            train_test_validate_split(0.7,0.2,0.1,X,y)
        X_train = enc.transform(X_train).toarray()
        X_test = enc.transform(X_test).toarray()
        X_val = enc.transform(X_val).toarray()
        dic = {'X_train': X_train,
            'X_test': X_test,
            'X_val': X_val,
            'y_train':y_train,
            'y_test':y_test,
            'y_val':y_val
            }
    return dic


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