import numpy as np
from ..activations import Activations


class BaseLayer:
    
    def __init__(self,name, layerType, nNeurons,activationType='None'):
        
        self.name = name
        self.layerType = layerType
        
        self.nNeurons = nNeurons
        
        self.nextLayer = 0
        self.previousLayer = 0
        
        self.activationType = activationType
        self.preActivation = 0
        self.activatedValues = 0
        self.nParameters = 0
        
    def __call__(self, previousLayer):
        if self.layerType == 'input':
            print "Error: Input layer cannot have any previous layer"
            return
        
        self.previousLayer= previousLayer
        self.previousLayer.nextLayer = self
        self.previousNeurons = self.previousLayer.nNeurons

        # weight.shape = [inputDim * outputDim]
        self.weights = np.random.normal(scale=1.0/np.sqrt(self.previousNeurons),size=(self.previousNeurons,self.nNeurons))
        self.bias = np.random.normal(size=(1,self.nNeurons))
        self.nParameters = self.weights.shape[0]*self.weights.shape[1] + self.bias.shape[1]
        
    @staticmethod        
    def __forwardPropagate__(self, inputX):
        if self.layerType == 'input':
            self.activatedValues = inputX
        else:
            self.preActivation = np.matmul(inputX,self.weights) + self.bias
            self.activatedValues = Activations.getActivatedValues(self.preActivation, self.activationType )            
        
        # output
        if self.layerType =='output':
            self.predictions = self.activatedValues 
            self.predictedLabels = np.argmax(self.activatedValues,1)

    @staticmethod        
    def __calculateGradient__(self):
        if self.layerType == 'input':
            print "Error: Input layer cannot have any gradient"
            return
        elif self.layerType == 'output':
            self.preActivationGradient = -(np.eye(self.nClass)[self.trueLabels] -self.activatedValues)
        else:
            outGradient = self.nextLayer.prevLayerGradient
            self.activationGradient = Activations.getActivationGradient(self.preActivation, self.activatedValues, self.activationType)
            self.preActivationGradient = self.activationGradient * outGradient
        
        batchSize = float(self.activatedValues.shape[0])
#         self.weightGradient = np.matmul(self.preActivationGradient.T,self.previousLayer.activatedValues )
        self.weightGradient = np.matmul(self.previousLayer.activatedValues.T, self.preActivationGradient)/batchSize
        self.biasGradient = np.mean(self.preActivationGradient, axis=0)
        self.prevLayerGradient = np.matmul(self.preActivationGradient, self.weights.T)



class Input(BaseLayer):
    def __init__(self, nFeatures = 100):
        BaseLayer.__init__(self,name="Input",layerType='input', nNeurons=nFeatures, activationType='None')
        
    def forwardPropagate(self, inputX):
        BaseLayer.__forwardPropagate__(self,inputX)

        
class Dense(BaseLayer):
    def __init__(self,nNeurons=100, activation='relu', name = 'layerX'):
        BaseLayer.__init__(self,name=name, layerType='dense', nNeurons=nNeurons, activationType=activation)
        
    def __call__(self, previousLayer):
        BaseLayer.__call__(self, previousLayer)
        return self
        
    def forwardPropagate(self, inputX ):
        BaseLayer.__forwardPropagate__(self, inputX)
        
    def calculateGradient(self):
        BaseLayer.__calculateGradient__(self)
        
        
class Output(BaseLayer):       
    def __init__(self,nClass=100, activation='softmax',name="Output"):
        if activation != 'softmax':
            print "Error: Output layer only supports softmax activation"
            return
        
        BaseLayer.__init__(self,name=name, layerType='output', nNeurons=nClass, activationType=activation)
        self.nClass = nClass
        self.trueLabels = 0
        
    def __call__(self, previousLayer):
        BaseLayer.__call__(self, previousLayer)
        return self
        
    def forwardPropagate(self, inputX ):
        BaseLayer.__forwardPropagate__(self, inputX)
    
    def calculateGradient(self, trueLabels):
        self.trueLabels = trueLabels
        BaseLayer.__calculateGradient__(self)