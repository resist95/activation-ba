
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import pickle
import gzip

# INTEL Datensatz aus
# Eigenschaften: 6 Kategorien aufgeteilt in train und test train 14034 Dateien 3000 test
# Bildgroesse 150 x 150 x 3
# 
# Dateiformat nach extraktion Label,Data -> ndarray


class Intel():
    
    def __init__(self):
        #Init des Dateipfads sowie laden der Daten
        self.path_to_dir = 'files/intel'
        self.data,self.labels = self.__load()
        
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
            resized = cv.resize(I,size)
            RGB_img = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            train_data.append(RGB_img)

        lb = LabelEncoder()
        train_labels = lb.fit_transform(train_labels)

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
            resized = cv.resize(I,size)
            RGB_img = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            test_data.append(RGB_img)

        lb = LabelEncoder()
        test_labels = lb.fit_transform(test_labels)
        data = train_data +test_data
        labels = np.append(train_labels,test_labels)
        data = np.array(data)
        
        return data,labels

    def get_data(self):
        return self.data,self.labels
   
    def print_data(self):      
        for i in range(1,9):
            plt.subplot(330+1*i)
            plt.imshow(self.data[i], cmap=plt.get_cmap('gray'))
            
        plt.show()
    
    def get_mean_std_deviation(self):
        # Berechnen vom mean und std abweichung der einzelnen pixelwerte
        mean = []
        std = []
        mean = self.data.mean(axis=(0,1))
        std = self.data.std(axis=(0,1))
        mean = np.mean(mean,axis=(0)) /255
        std = np.mean(std,axis=(0)) /255

        return np.round(mean,4), np.round(std,4)


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

    def __init__(self):
        self.path_to_dir = 'files/cifar-10-batches-py'
        self.meta_data = unpickle(self.path_to_dir + "/batches.meta")
        self.data,self.labels = self.__load()


    def __load(self):
        #num_cases_per_batch: 10000
        #label_names': airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
        #num_vis: 3072
        label_names = self.meta_data[b'label_names']
        label_names = np.array(label_names)
            
        data = []
        labels = []
       
        for i in range(1,6):
            #Data,filenames und label extrahieren
            curr_batch = unpickle(self.path_to_dir + "/data_batch_{}".format(i))
            #keys: b'batch_label', b'labels', b'data', b'filenames'
        
            data.append(curr_batch[b'data']) #shape: 10000,3072 -> 3,32,32
            
            labels += curr_batch[b'labels']
        
        cifar_test_batch = unpickle(self.path_to_dir + "/test_batch")
        data.append(cifar_test_batch[b'data'])
        labels += cifar_test_batch[b'labels']
        #Formatierung der Liste in 3,32,32 das nn mit den Daten arbeiten kann
        data = np.reshape(data,(len(data[0])*6,3072))
        data = data.reshape(len(data),3,32,32)
        data = np.rollaxis(data, 1, 4)

        return data,labels
    
    def get_data(self):
        return self.data,self.labels
           
    def print_data(self,train_data,name):

        plot, ax = plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
                idx = np.random.randint(0,self.data.shape[0])

                ax[i,j].imshow(train_data[idx])
                ax[i,j].set_xlabel(name[idx])
                ax[i,j].get_yaxis().set_visible(False)
        plt.show()

    def get_mean_std_deviation(self):
        mean = []
        std = []
        mean = self.data.mean(axis=(0,1))
        std = self.data.std(axis=(0,1))
        mean = np.mean(mean,axis=(0)) /255
        std = np.mean(std,axis=(0)) /255
        
        return np.round(mean,4), np.round(std,4)
    

# MNIST Datensatz aus http://yann.lecun.com/exdb/mnist/
# Eigenschaften: 60000 Dateien Trainingssatz und 10000 Testdatensaetze
# Bildgroesse: 28x28
# label_names: 1-9
#
#


class MNIST():

    def __init__(self):
        
        path_to_dir = 'files/mnist/raw'
        self.filename_train_data = path_to_dir+"/train-images-idx3-ubyte.gz"
        self.filename_train_labels = path_to_dir + "/train-labels-idx1-ubyte.gz"
        self.filename_test_data = path_to_dir + "/t10k-images-idx3-ubyte.gz"
        self.filename_test_labels = path_to_dir + "/t10k-labels-idx1-ubyte.gz"

        self.data,self.labels = self.__load()
    
    def __load(self):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        with gzip.open(self.filename_train_data, 'rb') as f:
        #TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        #[offset] [type]          [value]          [description]
        #0000     32 bit integer  0x00000803(2051) magic number
        #0004     32 bit integer  60000            number of images
        #0008     32 bit integer  28               number of rows
        #0012     32 bit integer  28               number of columns
        #0016     unsigned byte   ??               pixel
        #0017     unsigned byte   ??               pixel
        #........
        #xxxx     unsigned byte   ??               pixel
        #Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
           magic_number =  int.from_bytes(f.read(4),'big')
           image_count = int.from_bytes(f.read(4),'big')
           image_rows = int.from_bytes(f.read(4), 'big')
           image_cols = int.from_bytes(f.read(4),'big')
           image_data = f.read()

           train_data = np.frombuffer(image_data,dtype=np.uint8)
          

        #Train Labels entpacken
        with gzip.open(self.filename_train_labels,'rb') as f:
        #TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        #[offset] [type]          [value]          [description]
        #0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        #0004     32 bit integer  60000            number of items
        #0008     unsigned byte   ??               label
        #0009     unsigned byte   ??               label
        #........
        #xxxx     unsigned byte   ??               label
        #The labels values are 0 to 9.
            magic_number = int.from_bytes(f.read(4),'big')
            label_count = int.from_bytes(f.read(4), 'big')
            labels = f.read()
            train_labels = np.frombuffer(labels,dtype=np.uint8)

        with gzip.open(self.filename_test_data,'rb') as f:
        #TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
        #[offset] [type]          [value]          [description]
        #0000     32 bit integer  0x00000803(2051) magic number
        #0004     32 bit integer  10000            number of images
        #0008     32 bit integer  28               number of rows
        #0012     32 bit integer  28               number of columns
        #0016     unsigned byte   ??               pixel
        #0017     unsigned byte   ??               pixel
        #........
        #xxxx     unsigned byte   ??               pixel
        #Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
            magic_number =  int.from_bytes(f.read(4),'big')
            image_count_t = int.from_bytes(f.read(4),'big')
            image_rows = int.from_bytes(f.read(4), 'big')
            image_cols = int.from_bytes(f.read(4),'big')
            image_data = f.read()

            test_data = np.frombuffer(image_data,dtype=np.uint8)
            
        with gzip.open(self.filename_test_labels,'rb') as f:
        #TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
        #[offset] [type]          [value]          [description]
        #0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        #0004     32 bit integer  10000            number of items
        #0008     unsigned byte   ??               label
        #0009     unsigned byte   ??               label
        #........
        #xxxx     unsigned byte   ??               label
        #The labels values are 0 to 9.
            magic_number = int.from_bytes(f.read(4),'big')
            label_count = int.from_bytes(f.read(4), 'big')
            labels = f.read()
            test_labels = np.frombuffer(labels,dtype=np.uint8)
        # np.shape = (60000, 28, 28) train_data
        # (60000,) train_labels
        # (10000, 28, 28) test_data
        # (10000,) test_labels
        count = image_count+image_count_t
        data = np.append(train_data,test_data)
        data = data.reshape(count,image_rows,image_cols)

        labels = np.append(train_labels,test_labels)
        return data,labels
    
    def get_data(self):
        return self.data,self.labels

    def print_data(self):
            
        for i in range(1,9):
            plt.subplot(330+1*i)
            plt.imshow(self.data[i], cmap=plt.get_cmap('gray'))
            plt.show()
    
    def get_mean_std_deviation(self):
        mean = self.data.mean() / 255
        std_deviation = self.data.std() / 255
        return round(mean,4),round(std_deviation,4)


def main():
    cal = Intel()
    cal.get_mean_std_deviation()
    
if __name__== "__main__":

    main()