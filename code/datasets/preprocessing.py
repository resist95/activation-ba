from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import numpy as np

#aufsplitten in train und validation data
def train_validate_split(X_train,y_train,test_size=0):
    X_t, y_t = shuffle(X_train,y_train) #shuffle funktion von sklearn zur durchmischung von trainingsdaten fuer validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_t,y_t,test_size=test_size) #aufteilung in training und validation 
    return X_tr,y_tr,X_val,y_val

#shufflen train data, da nn mit gleichen daten arbeiten sollen wird shuffle von pytorch deaktiviert
#wirft value error bei test_size 0 deshalb extra funktion
def shuffle_train_data(X_train,y_train):
    X_t, y_t = shuffle(X_train,y_train)
    return X_t,y_t

#ohc um softmax zu verbessern
def one_hot_encoding(train_labels,test_labels,num_classes):
    lb = LabelEncoder()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    train_label_ohc = []
    test_label_ohc = []
    for l in train_labels:
            train_label_ohc.append(np.eye(num_classes)[l])
    for l in test_labels:
            test_label_ohc.append(np.eye(num_classes)[l])
    return train_label_ohc,test_label_ohc

