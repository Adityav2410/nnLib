import numpy as np


class Activations:
    @staticmethod
    def relu(x):
        return x*(x>0).astype(int)
    
    @staticmethod
    def sigmoid(x):
        return( 1.0/(1+np.exp(-x)) )
    
    @staticmethod
    def softmax(x):
        y = np.exp(x)
        y = np.divide(y, np.sum(y,1).reshape([-1,1]) )
        return y
    
    @staticmethod
    def getActivatedValues(x, activationType):
        if activationType == 'relu':
            return Activations.relu(x)
        elif activationType == 'sigmoid':
            return Activations.sigmoid(x)
        elif activationType == 'tanh':
            return Activations.tanh(x)
        elif activationType == 'softmax':
            return Activations.softmax(x)
        
    @staticmethod
    def grad_relu(x):
        return x>0
    
    @staticmethod
    def grad_sigmoid(y):
        return y*(1-y)
    
    
    @staticmethod
    def getActivationGradient(inputX ,outputY, activationType):
        if activationType == 'relu':
            return Activations.grad_relu(inputX)
        elif activationType == 'sigmoid':
            return Activations.grad_sigmoid(outputY)
        elif activationType =='tanh':
            return Activations.grad_tanh(outputY)