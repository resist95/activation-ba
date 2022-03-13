import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import glob

# CIFAR10 Datensatz aus https://www.cs.toronto.edu/~kriz/cifar.html
# Eigenschaften: 50000 Dateien aufgeteilt in 5 batches und testbatch mit 10000 Dateien
# Bildgroesse 32x32x3
# label_names: airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
#
# Dateiformat nach extraktion Label,Data -> ndarray 

# aus CIFAR10 Seite
# zum entpacken der Dateien
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10():

    def __init__(self,dir):
        self.meta_data = unpickle(dir + "/batches.meta")
        self.path_to_dir = dir
        self.cifar_train_data,self.cifar_train_labels,self.cifar_test_data,self.cifar_test_labels = self.__load_data()
    
    def get_cifar_data(self):
        return self.cifar_train_data,self.cifar_train_labels,self.cifar_test_data,self.cifar_test_labels

    def __load_data(self):
        #num_cases_per_batch: 10000
        #label_names': airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
        #num_vis: 3072
        label_names = self.meta_data[b'label_names']
        label_names = np.array(label_names)
        
        
        cifar_train_data = []
        cifar_train_labels = []

        cifar_test_data = []
        cifar_test_labels = []
       
        for i in range(1,6):
            #Data,filenames und label extrahieren
            curr_batch = unpickle(self.path_to_dir + "/data_batch_{}".format(i))
            #keys: b'batch_label', b'labels', b'data', b'filenames'
        
            cifar_train_data.append(curr_batch[b'data']) #shape: 10000,3072 -> 3,32,32
            
            cifar_train_labels += curr_batch[b'labels']
        
        #Formatierung der Liste in 3,32,32 das nn mit den Daten arbeiten kann
        cifar_train_data = np.reshape(cifar_train_data,(len(cifar_train_data[0])*5,3072))
        cifar_train_data = cifar_train_data.reshape(len(cifar_train_data),3,32,32)
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)

        cifar_test_batch = unpickle(self.path_to_dir + "/test_batch")
        cifar_test_data = cifar_test_batch[b'data']
        cifar_test_data = cifar_test_data.reshape(len(cifar_test_data),3,32,32)
        cifar_test_labels = cifar_test_batch[b'labels']

        return cifar_train_data,cifar_train_labels,cifar_test_data,cifar_test_labels

    def rename_labels(self,train_label,test_label):


        for i in range(len(train_label)):
            
            if train_label[i] == 0:
                train_label[i] = 'airplane'
            elif train_label[i] == 1:
                train_label[i] = 'automobile'
            elif train_label[i] == 2:
                train_label[i] = 'bird'
            elif train_label[i] == 3:
                train_label[i] = 'cat'
            elif train_label[i] == 4:
                train_label[i] = 'deer'
            elif train_label[i] == 5:
                train_label[i] = 'dog'
            elif train_label[i] == 6:
                train_label[i] = 'frog'
            elif train_label[i] == 7:
                train_label[i] = 'horse'
            elif train_label[i] == 8:
                train_label[i] = 'ship'
            elif train_label[i] == 9:
                train_label[i] = 'truck'
        for i in range(len(test_label)):
            
            if test_label[i] == 0:
                test_label[i] = 'airplane'
            elif test_label[i] == 1:
                test_label[i] = 'automobile'
            elif test_label[i] == 2:
                test_label[i] = 'bird'
            elif test_label[i] == 3:
                test_label[i] = 'cat'
            elif test_label[i] == 4:
                test_label[i] = 'deer'
            elif test_label[i] == 5:
                test_label[i] = 'dog'
            elif test_label[i] == 6:
                test_label[i] = 'frog'
            elif test_label[i] == 7:
                test_label[i] = 'horse'
            elif test_label[i] == 8:
                test_label[i] = 'ship'
            elif test_label[i] == 9:
                test_label[i] = 'truck'

        return train_label,test_label

    def print_data(self,train_data,name):

        plot, ax = plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
                idx = np.random.randint(0,train_data.shape[0])

                ax[i,j].imshow(train_data[idx])
                ax[i,j].set_xlabel(name[idx])
                ax[i,j].get_yaxis().set_visible(False)
        plt.show()



def main():
    cif = CIFAR10('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/cifar-10-batches-py/')
    cifar_train_data,cifar_train_labels = cif.get_cifar_data()
    cif.print_data(cifar_train_data,cifar_train_labels)


if __name__== "__main__":

    main()