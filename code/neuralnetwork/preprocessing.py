import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchvision import  transforms


def validate_split(test_ratio, validation_ratio, X, y):
    X_val,X_test,y_val,y_test = train_test_split(X,y,test_size=test_ratio / (test_ratio + validation_ratio))
    return X_val,y_val



