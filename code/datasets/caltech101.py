import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob

from sklearn.preprocessing import LabelEncoder
# CALTECH101 Datensatz aus http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# Eigenschaften: 101 Kategorien. 40 bis 800 Bilder pro Kategorie
# Bildgroesse 300 x 200 x 3
# 
# Dateiformat nach extraktion Label,Data -> ndarray


class caltech101():
    
    def __init__(self,path_to_dir):
        
        self.path_to_dir = path_to_dir
        self.caltech_train_data, self.caltech_labels = self.__load_data(path_to_dir)
        
    def __load_data(self,path_to_dir):
        path_to_img = path_to_dir + '/*/*.jpg'
        imgfiles = []
        for file in glob.glob(path_to_img):
            imgfiles.append(file)
        
        data = []
        labels = []

        for paths in imgfiles:
            labels.append(paths.split(os.path.sep)[-2])
            img = cv2.imread(paths)
            
            data.append(img)
        data = np.array(data,dtype=object)
        labels = np.array(labels,dtype=object)
        #Dublipkate aus labels list entfernen 
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        return data,labels

    
def main():
    cif = caltech101('E:/Explorer/Dokumente/Bachelor/git/activation-ba/files/caltech101/101_ObjectCategories')

    #cifar_train_data,cifar_train_labels,cifar_test_data,cifar_test_labels = cif.load_data()
    #cif.print_data(cifar_train_data,cifar_train_labels)


if __name__== "__main__":

    main()