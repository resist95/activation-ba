
import numpy as np
import gzip
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset

# MNIST Datensatz aus http://yann.lecun.com/exdb/mnist/
# Eigenschaften: 60000 Dateien Trainingssatz und 10000 Testdatensaetze
# Bildgroesse: 28x28
# label_names: 1-9
#
#


class MNIST(Dataset):

    def __init__(self,path_to_dir,transforms=None):
        
        self.filename_train_data = path_to_dir+"/train-images-idx3-ubyte.gz"
        self.filename_train_labels = path_to_dir + "/train-labels-idx1-ubyte.gz"
        self.filename_test_data = path_to_dir + "/t10k-images-idx3-ubyte.gz"
        self.filename_test_labels = path_to_dir + "/t10k-labels-idx1-ubyte.gz"

        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.train_data,self.train_labels,self.test_data,self.test_labels = \
        self.__load()
        self.transform = transforms
    
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
           train_data = train_data.reshape(image_count,image_rows,image_cols)

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
            image_count = int.from_bytes(f.read(4),'big')
            image_rows = int.from_bytes(f.read(4), 'big')
            image_cols = int.from_bytes(f.read(4),'big')
            image_data = f.read()

            test_data = np.frombuffer(image_data,dtype=np.uint8)
            test_data = test_data.reshape(image_count,image_rows,image_cols)
        
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

        return train_data,train_labels,test_data,test_labels
    
    def __len__(self):
        return len(self.train_labels),len(self.test_labels)

    def __getitem__(self,train=False):
        data_train = self.train_data
        data_test = self.test_data
        if train:
            if self.transform:
                data_train = self.transform(data_train)
                return data_train,self.train_labels
            return data_train,self.train_labels
        else: 
            if self.transform:
                data_test = self.transform(data_test)
                return data_test,self.test_labels
            return data_test, self.test_labels

    def print_data(self):
            
        for i in range(1,9):
            plt.subplot(330+1*i)
            plt.imshow(self.train_data[i], cmap=plt.get_cmap('gray'))
            plt.show()
    
    def get_mean_std_deviation(self):
        mean = self.train_data.mean() / 255
        std_deviation = self.train_data.std() / 255
    
        return round(mean,4),round(std_deviation,4)

def main():
    cif = MNIST('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/mnist/raw')



if __name__== "__main__":

    main()