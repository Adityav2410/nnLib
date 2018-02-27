import numpy as np

class Metrics:

    @staticmethod
    def crossEntropyLoss( trueLabels, predictions):
        return round(-np.mean( np.log(predictions[np.arange(len(predictions)), trueLabels]) ),5)
    
    @staticmethod
    def accuracy(trueLabels, predictedLabels, percent=False):
        acc = np.mean(trueLabels == predictedLabels)
        if percent:
        	return round(acc*100,2)
        else:
        	return acc
