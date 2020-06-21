#!/usr/bin/env python
# coding: utf-8

# In[65]:


from torchvision.datasets import VisionDataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image

import os
import os.path
import sys
import numpy as np
import pandas as pd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split 
        
        # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')
        
        path = "/content/Caltech101/" + split + ".txt"
        f = np.loadtxt(path, dtype = 'str')
        
        # image_path.split('/')[0] is the label ex. airplanes/image_0177.jpg 
        # create a DataFram with columns [image, label, encoded_label]
        self.data = pd.DataFrame([ [pil_loader('/content/Caltech101/101_ObjectCategories/' + image_path), image_path.split('/')[0] ] 
                                  for image_path in f if not image_path.startswith('BACKGROUND_Google')], columns = ['images', 'labels'])
        
        # LabelEncoder() encodes target labels with value between 0 and n_classes-1
        encoder = LabelEncoder()
        label_encoded = encoder.fit_transform(self.data['labels'])
        self.data['labels_encoded'] = label_encoded

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data.iloc[index]['images'], self.data.iloc[index]['labels_encoded']  
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length

