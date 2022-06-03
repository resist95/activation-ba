
# default dependencies
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import gzip
import torch


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
#special dependencies
from datasets.preprocessing import train_validate_split, one_hot_encoding,shuffle_train_data
from sklearn.preprocessing import LabelEncoder

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# INTEL Datensatz aus
# Eigenschaften: 6 Kategorien aufgeteilt in train und test train 14034 Dateien 3000 test
# Bildgroesse 150 x 150 x 3
# 
# Dateiformat nach extraktion Label,Data -> ndarray
class Intel():
    
    def __init__(self,test_size,run):
        #Init des Dateipfads sowie laden der Daten
        self.path_to_dir = 'files/intel'
        self.test_size = test_size
        self.run = run
        self.images, self.labels, self.test_images,self.test_labels = self.__load()
    
               
    def __load(self):
        # Laden der Daten mithilfe von glob das alle
        # Daten mit .jpg auflistet und liste hinzufügt
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
        train_data = np.reshape(train_data,(len(train_data),3,150,150))
        test_data = np.reshape(test_data,(len(test_data),3,150,150))

        return train_data, train_labels, test_data,test_labels

    #Split Daten in Test, val und training set
    def __prepare_data(self,train_data,train_labels,X_test,test_labels):
        train_lb,y_test = one_hot_encoding(train_labels,test_labels,6)

        if self.run == 'validate': #Falls Validation set gebraucht wird Split aus Train Daten generieren
            X_train,y_train,X_val,y_val = train_validate_split(train_data,train_lb,self.test_size)
        elif self.run == 'test':
            X_train,y_train = shuffle_train_data(train_data,train_lb)
            X_val = 0 
            y_val = 0
        self.dict_images = {'X_train':X_train,
                            'X_test': X_test,
                            'X_val': X_val,
                        }
        self.dict_labels = {'y_train': y_train,
                            'y_test': y_test,
                            'y_val': y_val
                        }
    
    def get_data(self,what_is_needed):
      #Abhängig welche Daten benötigt werden werden Dictionary Eintraege uebergeben
        if what_is_needed == 'train':
            return (self.dict_images['X_train'],self.dict_labels['y_train'])
        elif what_is_needed == 'test':
            return (self.dict_images['X_test'],self.dict_labels['y_test'])
        elif what_is_needed == 'val':
            return (self.dict_images['X_val'],self.dict_labels['y_val'])
        elif what_is_needed == 'all':
            return (self.dict_images,self.dict_labels)

    def print_data(self):
        self.X_train = np.reshape(self.X_train,(len(self.X_train),150,150,3))      
        for i in range(1,9):
            plt.subplot(330+1*i)
            plt.imshow(self.X_train[i], cmap=plt.get_cmap('gray'))
            
        plt.show()
    
    def get_mean_std(self):
      #mean und std berechnung zur standardisierung der Daten
        t = torch.from_numpy(self.dict_images['X_train']*1.0)
        means = t.mean(dim=0, keepdim=False) /255
        stds = t.std(dim=0,keepdim=False) /255
        mean = means
        std =  stds
        return mean,std
    
    def prepare_data(self):
        self.__prepare_data(self.images,self.labels,self.test_images,self.test_labels)

    #probably outdated
    def prepare_data_grad(self,batch_size):
      #Gradient benoetigt anderes Dictionary
        train_labels = self.labels
        train_images = self.images
        dat = []
        labels = []
        counter = [0,0,0,0,0,0]
        lb = LabelEncoder()
        train_labels = lb.fit_transform(train_labels)
        
        X_train, y_train = shuffle_train_data(train_images,train_labels)
        
        for i,data in enumerate(X_train):
            if y_train[i] == 0 and counter[0] < batch_size :
                dat.append(data)
                labels.append(y_train[i])
                counter[0] += 1
            elif y_train[i] == 1 and counter[1] < batch_size :
                dat.append(data)
                labels.append(y_train[i])
                counter[1] += 1
            elif y_train[i] == 2 and counter[2] < batch_size :
                dat.append(data)
                labels.append(y_train[i])
                counter[2] += 1
            elif y_train[i] == 3 and counter[3] < batch_size :
                dat.append(data)
                labels.append(y_train[i])
                counter[3] += 1
            elif y_train[i] == 4 and counter[4] < batch_size :
                dat.append(data)
                labels.append(y_train[i])
                counter[4] += 1
            elif y_train[i] == 5 and counter[5] < batch_size :
                dat.append(data)
                labels.append(y_train[i])
                counter[5] += 1
        y_train, y_test = one_hot_encoding(labels,self.test_labels,6)

        d = np.asarray(dat)
        self.dict_images = {'X_train': d,
                    'X_test' : self.test_images,
                    }
        self.dict_labels = {'y_train': y_train,
                    'y_test': y_test
                    }
    
    #probably outdated
    def prepare_data_act(self, batch_size):
      #Act benoetigt anderes Dictionary
        test_labels = self.test_labels
        test_images = self.test_images
        
        train_labels = self.labels
        train_images = self.images
        dat = []
        labels = []
        counter = [0,0,0,0,0,0]
        lb = LabelEncoder()
        train_labels = lb.fit_transform(train_labels)
        test_labels = lb.fit_transform(test_labels)
        X_train, y_train = shuffle_train_data(train_images,train_labels)
        
        for i,data in enumerate(test_images):
            if test_labels[i] == 0 and counter[0] < batch_size :
                dat.append(data)
                labels.append(test_labels[i])
                counter[0] += 1
            elif test_labels[i] == 1 and counter[1] < batch_size :
                dat.append(data)
                labels.append(test_labels[i])
                counter[1] += 1
            elif test_labels[i] == 2 and counter[2] < batch_size :
                dat.append(data)
                labels.append(test_labels[i])
                counter[2] += 1
            elif test_labels[i] == 3 and counter[3] < batch_size :
                dat.append(data)
                labels.append(test_labels[i])
                counter[3] += 1
            elif test_labels[i] == 4 and counter[4] < batch_size :
                dat.append(data)
                labels.append(test_labels[i])
                counter[4] += 1
            elif test_labels[i] == 5 and counter[5] < batch_size :
                dat.append(data)
                labels.append(test_labels[i])
                counter[5] += 1
        y_train, y_test = one_hot_encoding(train_labels,labels,6)

        d = np.asarray(dat)
        self.dict_images = {'X_train': X_train,
                    'X_test' : d,
                    }
        self.dict_labels = {'y_train': y_train,
                    'y_test': y_test
                    }


# CIFAR10 Datensatz aus https://www.cs.toronto.edu/~kriz/cifar.html
# Eigenschaften: 50000 Dateien aufgeteilt in 5 batches und testbatch mit 10000 Dateien
# Bildgroesse 32x32x3
# label_names: airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
#
# Dateiformat nach extraktion Label,Data -> ndarray 

# aus CIFAR10 Seite
# zum entpacken der Dateien

class CIFAR10():

    def __init__(self,test_size,run):
        self.test_size = test_size
        self.run = run
        self.path_to_dir = 'files/cifar-10-batches-py'
        self.meta_data = unpickle(self.path_to_dir + "/batches.meta")
        self.images, self.labels, self.test_images, self.test_labels = self.__load()
        
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
        train_data = data[0:50000]
        train_labels = labels[0:50000]
        test_data = data[50000:]
        test_labels = labels[50000:]
        return train_data,train_labels,test_data,test_labels
    
    def __prepare_data(self,train_data,train_labels,X_test,test_labels):
      #siehe intel
        train_lb,y_test = one_hot_encoding(train_labels,test_labels,10)
        if self.run == 'validate':
            X_train,y_train,X_val,y_val = train_validate_split(train_data,train_lb,self.test_size)
        elif self.run == 'test':
            X_train,y_train= shuffle_train_data(train_data,train_lb)
            X_val = 0 
            y_val = 0
        self.dict_images = {'X_train':X_train,
                            'X_test': X_test,
                            'X_val': X_val,
                        }
        self.dict_labels = {'y_train': y_train,
                            'y_test': y_test,
                            'y_val': y_val
                        }

    def get_data(self,what_is_needed):
      #siehe intel
        if what_is_needed == 'train':
            return (self.dict_images['X_train'],self.dict_labels['y_train'])
        elif what_is_needed == 'test':
            return (self.dict_images['X_test'],self.dict_labels['y_test'])
        elif what_is_needed == 'val':
            return (self.dict_images['X_val'],self.dict_labels['y_val'])
        elif what_is_needed == 'all':
            return (self.dict_images,self.dict_labels)
          
    def print_data(self):

        plot, ax = plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
                idx = np.random.randint(0,self.dict_images['X_train'].shape[0])

                ax[i,j].imshow(self.dict_images['X_train'][idx])
                ax[i,j].set_xlabel(self.dict_labels['y_train'][idx])
                ax[i,j].get_yaxis().set_visible(False)
        plt.show()

    def get_mean_std(self):
      #siehe intel
        mean = []
        std = []
        t = torch.from_numpy(self.dict_images['X_train']*1.0)
        means = t.mean(dim=0, keepdim=False) /255
        stds = t.std(dim=0,keepdim=False) /255
        mean = means
        std =  stds
        return mean,std
    
    def prepare_data(self):
      self.__prepare_data(self.images,self.labels,self.test_images,self.test_labels)
    
    #probably outdated
    def prepare_data_grad(self,batch_size):
      #siehe intel
      train_labels = self.labels
      train_images = self.images
      dat = []
      labels = []
      counter = [0,0,0,0,0,0,0,0,0,0]
      lb = LabelEncoder()
      train_labels = lb.fit_transform(train_labels)
      
      X_train, y_train = shuffle_train_data(train_images,train_labels)
      
      for i,data in enumerate(X_train):
        if y_train[i] == 0 and counter[0] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[0] += 1
        elif y_train[i] == 1 and counter[1] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[1] += 1
        elif y_train[i] == 2 and counter[2] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[2] += 1
        elif y_train[i] == 3 and counter[3] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[3] += 1
        elif y_train[i] == 4 and counter[4] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[4] += 1
        elif y_train[i] == 5 and counter[5] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[5] += 1
        elif y_train[i] == 6 and counter[6] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[6] += 1
        elif y_train[i] == 7 and counter[7] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[7] += 1
        elif y_train[i] == 8 and counter[8] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[8] += 1
        elif y_train[i] == 9 and counter[9] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[9] += 1
      y_train, y_test = one_hot_encoding(labels,self.test_labels,10)

      d = np.asarray(dat)
      self.dict_images = {'X_train': d,
                    'X_test' : self.test_images,
                    }
      self.dict_labels = {'y_train': y_train,
                    'y_test': y_test
                    }
    
    #probably outdated
    def prepare_data_act(self,batch_size):
      #siehe intel
      test_labels = self.test_labels
      test_images = self.test_images
      
      train_labels = self.labels
      train_images = self.images
      dat = []
      labels = []
      counter = [0,0,0,0,0,0,0,0,0,0]
      lb = LabelEncoder()
      train_labels = lb.fit_transform(train_labels)
      test_labels = lb.fit_transform(test_labels)
      X_train, y_train = shuffle_train_data(train_images,train_labels)
      
      for i,data in enumerate(test_images):
        if test_labels[i] == 0 and counter[0] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[0] += 1
        elif test_labels[i] == 1 and counter[1] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[1] += 1
        elif test_labels[i] == 2 and counter[2] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[2] += 1
        elif test_labels[i] == 3 and counter[3] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[3] += 1
        elif test_labels[i] == 4 and counter[4] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[4] += 1
        elif test_labels[i] == 5 and counter[5] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[5] += 1
        elif test_labels[i] == 6 and counter[6] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[6] += 1
        elif test_labels[i] == 7 and counter[7] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[7] += 1
        elif test_labels[i] == 8 and counter[8] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[8] += 1
        elif test_labels[i] == 9 and counter[9] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[9] += 1
      y_train, y_test = one_hot_encoding(train_labels,labels,10)

      d = np.asarray(dat)
      self.dict_images = {'X_train': X_train,
                    'X_test' : d,
                    }
      self.dict_labels = {'y_train': y_train,
                    'y_test': y_test
                    }


# MNIST Datensatz aus http://yann.lecun.com/exdb/mnist/
# Eigenschaften: 60000 Dateien Trainingssatz und 10000 Testdatensaetze
# Bildgroesse: 28x28
# label_names: 1-9
#
#
class MNIST():

    def __init__(self,test_size,run):
        
        path_to_dir = 'files/mnist/MNIST/raw'
        self.filename_train_data = path_to_dir+"/train-images-idx3-ubyte.gz"
        self.filename_train_labels = path_to_dir + "/train-labels-idx1-ubyte.gz"
        self.filename_test_data = path_to_dir + "/t10k-images-idx3-ubyte.gz"
        self.filename_test_labels = path_to_dir + "/t10k-labels-idx1-ubyte.gz"
        
        self.run = run
        self.test_size = test_size
        self.images, self.labels, self.test_images, self.test_labels = self.__load()
        

    def __load(self):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        labels_ohc = []
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
        data = data.reshape(count,1,image_rows,image_cols)
        labels = np.append(train_labels,test_labels)

        return data[:60000],labels[:60000],data[60000:],labels[60000:]
    
    def __prepare_data(self,train_data,train_labels,X_test,test_labels):
      #siehe intel
        train_lb,y_test = one_hot_encoding(train_labels,test_labels,10)
        if self.run == 'validate':
            X_train,y_train,X_val,y_val = train_validate_split(train_data,train_lb,self.test_size)
        elif self.run == 'test':
            X_train,y_train = shuffle_train_data(train_data,train_lb)
            X_val = 0 
            y_val = 0

        self.dict_images = {'X_train':X_train,
                            'X_test': X_test,
                            'X_val': X_val,
                        }
        self.dict_labels = {'y_train': y_train,
                            'y_test': y_test,
                            'y_val': y_val
                        }

    def get_data(self,what_is_needed):
      #siehe intel
        if what_is_needed == 'train':
            return (self.dict_images['X_train'],self.dict_labels['y_train'])
        elif what_is_needed == 'test':
            return (self.dict_images['X_test'],self.dict_labels['y_test'])
        elif what_is_needed == 'val':
            return (self.dict_images['X_val'],self.dict_labels['y_val'])
        elif what_is_needed == 'all':
            return (self.dict_images,self.dict_labels)

    def print_data(self):
            
        for i in range(1,9):
            plt.subplot(330+1*i)
            plt.imshow(self.data[i], cmap=plt.get_cmap('gray'))
            plt.show()

    def get_mean_std(self):
      #siehe intel
        mean = []
        std = []
        t = torch.from_numpy(self.dict_images['X_train']*1.0)
        means = t.mean(dim=0,keepdim=False) /255
        stds = t.std(dim=(0,1,2),keepdim=False) /255
        mean = means
        std = torch.sum(stds) /28
        return mean, std
    
    def prepare_data(self):
      self.__prepare_data(self.images,self.labels,self.test_images,self.test_labels)
    
    #probably outdated
    def prepare_data_grad(self,batch_size):
      #siehe intel
      train_labels = self.labels
      train_images = self.images
      dat = []
      labels = []
      counter = [0,0,0,0,0,0,0,0,0,0]
      lb = LabelEncoder()
      train_labels = lb.fit_transform(train_labels)
      
      X_train, y_train = shuffle_train_data(train_images,train_labels)
      
      for i,data in enumerate(X_train):
        if y_train[i] == 0 and counter[0] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[0] += 1
        elif y_train[i] == 1 and counter[1] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[1] += 1
        elif y_train[i] == 2 and counter[2] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[2] += 1
        elif y_train[i] == 3 and counter[3] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[3] += 1
        elif y_train[i] == 4 and counter[4] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[4] += 1
        elif y_train[i] == 5 and counter[5] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[5] += 1
        elif y_train[i] == 6 and counter[6] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[6] += 1
        elif y_train[i] == 7 and counter[7] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[7] += 1
        elif y_train[i] == 8 and counter[8] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[8] += 1
        elif y_train[i] == 9 and counter[9] < batch_size :
          dat.append(data)
          labels.append(y_train[i])
          counter[9] += 1
      y_train, y_test = one_hot_encoding(labels,self.test_labels,10)

      d = np.asarray(dat)
      self.dict_images = {'X_train': d,
                    'X_test' : self.test_images,
                    }
      self.dict_labels = {'y_train': y_train,
                    'y_test': y_test
                    }
    
    #probably outdated
    def prepare_data_act(self,batch_size):
      #siehe intel
      test_labels = self.test_labels
      test_images = self.test_images
      
      train_labels = self.labels
      train_images = self.images
      dat = []
      labels = []
      counter = [0,0,0,0,0,0,0,0,0,0]
      lb = LabelEncoder()
      train_labels = lb.fit_transform(train_labels)
      test_labels = lb.fit_transform(test_labels)
      X_train, y_train = shuffle_train_data(train_images,train_labels)
      
      for i,data in enumerate(test_images):
        if test_labels[i] == 0 and counter[0] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[0] += 1
        elif test_labels[i] == 1 and counter[1] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[1] += 1
        elif test_labels[i] == 2 and counter[2] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[2] += 1
        elif test_labels[i] == 3 and counter[3] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[3] += 1
        elif test_labels[i] == 4 and counter[4] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[4] += 1
        elif test_labels[i] == 5 and counter[5] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[5] += 1
        elif test_labels[i] == 6 and counter[6] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[6] += 1
        elif test_labels[i] == 7 and counter[7] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[7] += 1
        elif test_labels[i] == 8 and counter[8] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[8] += 1
        elif test_labels[i] == 9 and counter[9] < batch_size :
          dat.append(data)
          labels.append(test_labels[i])
          counter[9] += 1
      y_train, y_test = one_hot_encoding(train_labels,labels,10)

      d = np.asarray(dat)
      self.dict_images = {'X_train': X_train,
                    'X_test' : d,
                    }
      self.dict_labels = {'y_train': y_train,
                    'y_test': y_test
                    }

def main():
    cal = Intel(0.1,'test')
    cal.prepare_data_grad()
if __name__== "__main__":

    main()
