# from layers import Input, Dense, Output
import numpy as np
from metrics import Metrics
import pickle

class Model():
    def __init__(self, inputLayer, outputLayer, name = 'Model' ):
        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.modelName = name
        self.runForwardProp = False
        self.learning_Rate = 0.001
        self.optimizer = 'SGD'
        self.loss = 'categorical_crossEntropy'
        self.organizeLayers()
        
    def compileModel(self, learning_Rate=0.001, loss = 'categorical_crossEntropy', optimizer='SGD', regularizer='L2', lam1=0.001, lam2 = 0.001):
        self.learning_Rate = learning_Rate
        self.optimizer = optimizer
        self.loss = loss
        

    def organizeLayers(self):
        currentLayer = self.inputLayer
        self.layerList = []
        self.layerDict = {}
        self.nParameters = 0
        while(True):
            self.layerList.append(currentLayer)
            self.layerDict[currentLayer.name] = currentLayer
            self.nParameters += currentLayer.nParameters
            if currentLayer.layerType == 'output':
                break;
            currentLayer = currentLayer.nextLayer
#         dummyData = np.array([self.inputLayer.batchSize, self.inputLayer.nNeurons])
#         self.forwardPropagation(dummyData)
    
    def getLayer(self, layerName ):
        return self.layerDict[layerName]
        
    def summary(self, verbose=False):
        print "--------------- Model Summary -------------------"
        for i,currentLayer in enumerate(self.layerList):
            
            if currentLayer.layerType == 'input' or not verbose:
                print str(i+1)+".\t "+currentLayer.name+"\t\t\t\t "+str(currentLayer.nNeurons) 
            else:
                print str(i+1)+".\t "+currentLayer.name+"\t\t"+str(currentLayer.nNeurons) + "\tWeight Dimension:\t" + str(currentLayer.weights.shape) + "\tBias Dimension:\t" + str(currentLayer.bias.shape)
        print "\nTotal Number of parameters in model: \t"+str(self.nParameters)       

        
    def forwardPropagation(self, inputX):
        self.runForwardProp = True
        currentLayer = self.inputLayer
        while(True):
            if currentLayer.layerType == 'input':
                currentLayer.forwardPropagate(inputX)
            else:
                currentLayer.forwardPropagate(currentLayer.previousLayer.activatedValues)
            if currentLayer.layerType == 'output':
                break;
            currentLayer = currentLayer.nextLayer
        
    def backPropagation(self, trueLabels):
        for i in range(len(self.layerList)-1,0,-1):
            currentLayer = self.layerList[i]
            if currentLayer.layerType == 'output':
                currentLayer.calculateGradient(trueLabels)
            else:
                currentLayer.calculateGradient()
        
    def updateParameters(self):
        for i in range(len(self.layerList)-1,0,-1):
            currentLayer = self.layerList[i]
            if self.optimizer == 'SGD':
                currentLayer.weights = currentLayer.weights- self.learning_Rate*currentLayer.weightGradient
                currentLayer.bias = currentLayer.bias - self.learning_Rate*currentLayer.biasGradient
    
    def predict(self, trainX, labels = False):
        self.forwardPropagation(trainX)
        if labels:
            return self.outputLayer.predictedLabels
        else:
            return self.outputLayer.predictions
    
    def train(self, trainX, trainY, verbose = False ):
        trainY = np.squeeze(trainY)
        self.forwardPropagation(trainX)
        self.backPropagation(trainY)
        self.updateParameters()
        if verbose:
            loss = Metrics.crossEntropyLoss(trainY, self.outputLayer.predictions)
            accuracy = Metrics.accuracy(trainY, self.outputLayer.predictedLabels )
            return [loss, accuracy]

    def save(self, fileName):
        print "Saving model to file "+fileName
        pickle.dump(self, open(fileName,'wb'))
        print "Model saved"

    @staticmethod
    def load(fileName):
        loaded_model = pickle.load(open(fileName,'rb'))
        return loaded_model
