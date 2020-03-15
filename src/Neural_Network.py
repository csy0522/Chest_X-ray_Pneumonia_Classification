# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 02:46:21 2020

@author: CSY
"""


from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


'''
This class creates a neural network model
with customizable layers.
'''
class Neural_Network:
    
    def __init__(self,input_shape,output_shape,epochs,batch_size,layers):
        self.input_shape_ = input_shape
        self.output_shape_ = output_shape
        self.epochs_ = epochs
        self.batch_size_ = batch_size
        self.layers_ = layers
        self.model_ = self.__build_model__()


    '''
    The layers parameters will be used in this function.
    A sequential of layers will be added to the model
    '''
    def __build_model__(self):
        
        model = Sequential()
        for layer in self.layers_:
            model.add(layer)
        
        model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        
        return model

    
    '''
    This function trains the model
    '''
    def __train__(self,X,y,val_X,val_y):
        self.training_info_ = self.model_.fit(x=X, y=y, 
                        batch_size=self.batch_size_, epochs=self.epochs_, validation_split=1, 
                        validation_data=(val_X,val_y), shuffle=True)
        
        self.__plot__(self.training_info_.history['accuracy'])
        
        
    '''
    This function plots the training progress to a graph
    If the parameter 'savefig' is changed to True,
    the graph will be saved to a png file in current directory
    '''
    def __plot__(self, accuracy, savefig=False):
        plt.figure(figsize=(10,8))
        plt.title("Training_Progress")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(accuracy, label="Accuracy")
        plt.legend()
        if savefig == "True":
            plt.savefig("training_progress.png")
        plt.show()
        
        
    '''
    This function tests the trained model.
    It is recommended to run it after __train__
    '''
    def __test__(self,X,y):
        self.test_info_ = self.model_.evaluate(X,y,verbose=0)
        print("========================= MODEL TEST =========================")
        print("Loss test: {}".format(self.test_info_[0]))
        print("Accuracy test: {}".format(self.test_info_[1]))
        print("============================ END ============================")
        
        
    '''
    This function takes lung X-ray image as input
    and predict the condition.
    If the parameter 'output' is changed to True,
    The predicted result will be outputed.
    '''
    def __predict__(self, X, y, output=False):
        self.predict_info_ = self.model_.predict(X)
        if output == True:
            self.__plot_prediction__(X,y)
                
                
    '''
    This function outputs the prediction made from __predict__.
    It shows the image, prediction, and actual value as one line.
    If the input for prediction is an array, the output will be a chart.
    '''
    def __plot_prediction__(self,X,y):
        target_name = ["Normal", "Bacteria", "Virus"]
        predict = np.argmax(self.predict_info_, axis=1)
        predict = self.__convert_target__(predict,target_name)
        y = self.__convert_target__(y,target_name)
        num_data = len(self.predict_info_)
        plt.figure(figsize=(4,num_data),dpi=100)
        offset = 1
        self.predict_error_ = 0
        for i in range(num_data):
            sub = plt.subplot(num_data,2,i+offset)
            plt.imshow(X[i])
            sub.set_axis_off()
            
            sub2 = plt.subplot(num_data,2,i+offset+1)
            pred = np.argmax(self.predict_info_[i]) - y[i]
            color = "green"
            if pred != 0:
                self.predict_error_ += 1
                color = "red"
            plt.text(0.0,0.5,"predict: %s" % (predict[i]),bbox=dict(facecolor=color,alpha=0.5))
            plt.text(1.0,0.5,"Actual: %s" % (y[i]),bbox=dict(facecolor=color,alpha=0.5))
            sub2.set_axis_off()
            offset += 1
        plt.show()
        
        print("========================= Prediction Accuracy =========================")
        print("Correct Prediction: {}/{}".format(num_data-self.predict_error_),num_data)
        print("Incorrect Prediction: {}/{}".format(self.predict_error_),num_data)
        print("Accuracy: {}".format((num_data-self.predict_error_)/num_data))
        print("================================== END ==================================")
        
        
    '''
    This function converts the target values (integers) into specified names.
    The parameters 'names' contains a list of names that the user wishes to 
    use for target.
    '''    
    def __convert_target__(self,y,names):
        target = []
        for i in y:
            target.append(names[i])
        return target
    
    
    
    
    