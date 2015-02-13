'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618
         
'''

import os
import numpy as np
import operator
import scipy.spatial.distance as ssd

if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
featurepath = os.path.abspath(os.path.join(basepath, "Data5/proteins.features"))
labelpath = os.path.abspath(os.path.join(basepath, "Data5/proteins.labels"))

x = np.genfromtxt(featurepath, delimiter=',')
y = np.genfromtxt(labelpath, delimiter=',')

def filterTrainSet(testInstance,index):
    trainIndexes = []
    lowerIndex = index - 20
    upperIndex = index + 20
    trainIndexes.append(index)
    
    while lowerIndex > -1:
        trainIndexes.append(lowerIndex)
        lowerIndex = lowerIndex - 20
        
    while upperIndex < 400:
        trainIndexes.append(upperIndex)
        upperIndex = upperIndex + 20
        
    lowerBound = (index/10)*10
    
    if not lowerBound % 20 == 0:
        lowerBound = lowerBound - 10
       
    upperBound = lowerBound + 20
    
    for i in range(lowerBound,upperBound):
        trainIndexes.append(i)
        
    for i in range((index%10)*20,(index%10)*20+20):
        trainIndexes.append(i)
        
    a = lowerBound / 20
    
    while a < 400:
        trainIndexes.append(a)
        a = a + 20
    

    
    trainSet = []
    trainLabels = []
    for i in range(len(x)):
        if not i in trainIndexes:
            trainSet.append(x[i])
            trainLabels.append(y[i])
            
    return trainSet,trainLabels
    

def LooCV(modified):
    yPredictions = []
    for i in range(len(x)):
        if modified:
            trainSet,trainLabels = filterTrainSet(x[i],i)
            trainSet.append(x[i])
            trainLabels.append(y[i])
        else:
            trainSet = x
            trainLabels = y
        yPredictions.append(inferNeighbors(trainSet,x[i],trainLabels))
    
    return yPredictions
    
        

def inferNeighbors(trainSet,testInstance,labels):   
    distances = []
    for i in range(len(trainSet)):
        distances.append((ssd.euclidean(trainSet[i], testInstance), labels[i]))
        
    distances.sort(key=operator.itemgetter(0))    
    return distances[1][1]

def calculateCIndex(predictions, labels):
    n = 0
    h_sum = 0
    for i in range(len(labels)):
        t = labels[i]
        p = predictions[i]
        for j in range(i+1,len(labels)):
            nt = labels[j]
            np = predictions[j]
            if t != nt:
                n = n + 1
                if (p < np and t < nt) or (p > np and t > nt):
                    h_sum = h_sum + 1
                elif (p < np and t > nt) or (p > np and t < nt):
                    h_sum = h_sum + 0
                elif (p == np):
                    h_sum = h_sum + 0.5
                    
    if n == 0:
        return 0
    else:
        return h_sum/n

def main():
    predictedLabels = LooCV(False)
    print 'Concordance index of unmodified CV'
    print calculateCIndex(predictedLabels, y)
    print
    predictedLabels = LooCV(True)
    print 'Concordance index of modified CV'
    print calculateCIndex(predictedLabels, y)
    
main()
    