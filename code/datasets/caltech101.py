import numpy as np
import os

from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2 as cv

# CALTECH101 Datensatz aus http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# Eigenschaften: 101 Kategorien. 40 bis 800 Bilder pro Kategorie
# Bildgroesse 200 x 200 x 3
# 
# Dateiformat nach extraktion Label,Data -> ndarray


class caltech101(Dataset):
    
    def __init__(self,path_to_dir,transforms=None):
        #Init des Dateipfads sowie laden der Daten
        self.path_to_dir = path_to_dir
        self.data,self.labels = self.__load()
        self.transform = transforms
    
    def __load(self):
        # Laden der Daten mithilfe von glob das alle
        # Daten mit .jpg auflistet und liste hinzufügt
        path_to_img = self.path_to_dir + '/*/*.jpg'
        imgfiles = []
        for file in glob.glob(path_to_img):
            imgfiles.append(file)
        
        data = []
        labels = []
        size=(200,200)
        for paths in imgfiles:
        # Labels an Liste hinzufügen
        # Bilder resizen das alle gleiche groesse und dim haben
        # Zum Schluss umwandeln liste in ndarray
        # np.shape = (8676, 200, 200, 3)    
            labels.append(paths.split(os.path.sep)[-2])
            I = cv.imread(paths)
            img = cv.resize(I,size)
            data.append(img)
        data = np.asarray(data)
        print(np.shape(data))
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        return data,labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self):
        data = self.data
        if self.transform:
            data = self.transform(data)    
        return data,self.labels
    
    def print_data(self):      
        for i in range(1,9):
            plt.subplot(330+1*i)
            plt.imshow(self.train_data[i], cmap=plt.get_cmap('gray'))
            plt.show()
    
    def get_mean_std_deviation(self):
        # Berechnen vom mean und std abweichung der einzelnen pixelwerte
        mean = 0
        std_deviation = 0
        mean = self.data.mean() /255
        std_deviation = self.data.std() / 255

        return round(mean,4),round(std_deviation,4)
    
def main():
    cif = caltech101('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/caltech101/101_ObjectCategories')
    a,b = cif.get_mean_std_deviation()
    #cifar_train_data,cifar_train_labels,cifar_test_data,cifar_test_labels = cif.load_data()
    #cif.print_data(cifar_train_data,cifar_train_labels)


if __name__== "__main__":

    main()