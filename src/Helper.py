# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 02:46:59 2020

@author: CSY
"""


import cv2 as cv
import os
import numpy as np


'''
This class consists of helper functions to create neural network models.
'''
class Helper:
    
    def __init__(self):
        pass

    
    '''
    This function returns a set of images and its label.
    The parameter 't' specifies data's directories, which include
    "train", "validation", "test" ,"sample", and "predict".
    '''
    def _get_data_(self, t, shape, dimension, path="..\\data"):
        path += "\\%s" % (t)
        normal_path = path + "\\NORMAL\\"
        pneumonia_path = path + "\\PNEUMONIA\\"
        
        X = []
        y = []
        
        for c in os.listdir(normal_path):
            X.append(self._image_processing_(
            normal_path+c,shape,dim=dimension))
            y.append(0)
        for c in os.listdir(pneumonia_path):
            if "bacteria" in c:   
                X.append(self._image_processing_(
                pneumonia_path+c,shape,dim=dimension))
                y.append(1)
            elif "virus" in c:
                X.append(self._image_processing_(
                pneumonia_path+c,shape,dim=dimension))
                y.append(2)

        return np.array(X), np.array(y)
            
            
    '''
    This function processes an image (input) to fit the model's input shape.
    It returns either a 2d image or a 3d image.
    '''
    def _image_processing_(self, img_path, resize_shape, dim=2):
        img = cv.imread(img_path)
        if dim == 2:
            img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_resize = cv.resize(img,resize_shape)
        img_norm = img_resize / 255
        return img_norm
    
    
    '''
    This function already exists as part of Neural_Network class.
    It is also added to Helper class just in case needed.
    '''
    def _convert_target_(y, names):
        target = []
        for i in y:
            target.append(names[i])
        return target
        