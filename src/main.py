# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 02:52:39 2020

@author: CSY
"""

import keras.layers as kl
from Neural_Network import Neural_Network
from Helper import Helper



def main():
    # Helper Class
    h = Helper()


    # 2d Data
    X,y = h._get_data_('train',(60,60),2)
    val_X,val_y = h._get_data_('validation',(60,60),2)
    test_X,test_y = h._get_data_('test',(60,60),2)
    predict_X,predict_y = h._get_data_('predict',(60,60),2)
    
    
    # 3d Data
    X_3d,y_3d = h._get_data_('train',(60,60),3)
    val_X_3d,val_y_3d = h._get_data_('validation',(60,60),3)
    test_X_3d,test_y_3d = h._get_data_('test',(60,60),3)
    predict_X_3d,predict_y_3d = h._get_data_('predict',(60,60),3)
    
    
    
    # The layer for Artificial Neural Network (ANN) model.
    hidden_layers = [
        kl.Flatten(),
        kl.Dense(256),
        kl.Activation('sigmoid'),
        kl.Dense(64),
        kl.Activation('sigmoid'),
        kl.Dense(3,activation='softmax')
    ]

    # The layer for Convolutional Neural Network (CNN) model.
    conv_layers = [
        kl.Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),padding='same',input_shape=(60,60,3)),
        kl.Activation('relu'),
        kl.Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),padding='same'),
        kl.Activation('relu'),
        kl.Conv2D(filters=64,kernel_size=(2,2),strides=(1,1),padding='same'),
        kl.Activation('relu'),
        kl.Flatten(),
        kl.Dense(256),
        kl.Activation('relu'),
        kl.Dense(32),
        kl.Activation('relu'),
        kl.Dense(3,activation='softmax')
    ]
    
    

    # Artificial Neural Network (ANN) model
    ann = Neural_Network(input_shape=(60,60),output_shape=3,epochs=150,batch_size=200,layers=hidden_layers)
    
    # Convolutional Neural Network (CNN) model
    cnn = Neural_Network(input_shape=(60,60,3),output_shape=3,epochs=25,batch_size=200,layers=conv_layers)


    
    
    '''
    Train Models with Accuracy Graph
    '''
    # ann.__train__(X,y,val_X,val_y)
    # cnn.__train__(X_3d,y_3d,val_X_3d,val_y_3d)


    '''
    Test Models
    '''
    # ann.__test__(test_X,test_y)
    # cnn.__test__(test_X_3d,test_y_3d)


    '''
    Predict
    '''
    # ann.__predict__(predict_X,predict_y,output=True)
    # cnn.__predict__(predict_X_3d,predict_y_3d,output=True)




if __name__ == "__main__":
    
    main()







