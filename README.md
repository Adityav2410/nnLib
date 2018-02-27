## This is a toy library for building multi-layer perceptron. 
## Limitations:
 Only supports SGD optimizer
 Only supports Multilayer perceptron

## Getting started with model
```
import numpy as np
from nnLib.layers import Input, Dense, Output
from nnLib.model import Model
```


### Build model
```
def getModel():
    inputLayer = Input(nFeatures = 784 )
    dense1 = Dense(250, 'relu', name='Dense1')(inputLayer)
    # dense2 = Dense(128,'relu',name='Dense2')(dense1)
    outputLayer = Output(nClass = 10)(dense1)

    model = Model(inputLayer, outputLayer)
    model.compileModel(learning_Rate = 0.001)
    model.summary(verbose=True)
    return model
```

### Train the model 
```
model.train(trainX, trainY)
```

function accepts one batch of trainX, and trainY and fits over that data. 
trainY is the label and not one-hot vector

```
if __name__== "__main__":
    model = getModel()
    accuracyHistory = []
    lossHistory = []
    for i in range(5000):
        trainX, trainY = dataHandler.getTrainingBatch(1000)
        trainX = (trainX - 128)/128.0
        loss, accuracy = model.train(trainX, trainY, verbose = True)
        if i%50 == 0:
            print "Iter: "+str(i+1)+"\t\t Loss: " + str(round(loss,5)) + "\t\t Accuracy: "+str(round(100*accuracy,2))
        lossHistory.append(loss)
        accuracyHistory.append(accuracy)
```