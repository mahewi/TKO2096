'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618
         

Task steps: 

1. Implement a Leave-One-Out cross validation with deadzone radius R = 0, 10, 20, ..., 200. So you will do 21 analyses in total here. 
    Use 5-nearest neighbor as the prediction method. Remember normalization.  

2. Calculate the C-index value for each of the deadzone radius cases.

3. Plot the C-index vs. Deadzone radius in a graph to visualize, how the prediction performance changes with the deadzone radius. 
    Set Y-axis to be the C-index and X-axis to be Deadzone radius

4. Return your implementation and the graph in a written report. 

'''

import os
import operator
import scipy.spatial.distance as ssd
import numpy as np


if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
inputpath = os.path.abspath(os.path.join(basepath, "Data4/INPUT.csv"))
outputpath = os.path.abspath(os.path.join(basepath, "Data4/OUTPUT.csv"))
coordinatespath = os.path.abspath(os.path.join(basepath, "Data4/COORDINATES.csv"))

x = np.genfromtxt(inputpath, delimiter=',')
y = np.genfromtxt(outputpath, delimiter=',')
z = np.genfromtxt(coordinatespath, delimiter=',')


def calculateCIndex(predictions,index,labels):
    n = 0
    h_sum = 0
    for i in range(len(labels)):
        t = labels[i][index]
        p = predictions[i][index]
        for j in range(i+1,len(labels)):
            nt = labels[j][index]
            np = predictions[j][index]
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

def LooCV(k):
    yPredictions = []
    stdX = calculateZScore()
    for i in range(len(stdX)):
        neighbors = inferNeighbors(stdX,stdX[i],y,k,1)
        yPredictions.append(chooseMajorityLabel(neighbors,k))
        
    cIndex = calculateCIndex(yPredictions,0,y)
    print "Leave-one-out cross-validation"
    printCIndexes(cIndex)

def chooseMajorityLabel(neighbors,k):
    predictedOutcome = []
    for i in range(3):
        sumOfMod = 0.0
        for j in range(len(neighbors)):
            sumOfMod = sumOfMod + neighbors[j][1][i]
        predictedOutcome.append(sumOfMod/k)
    
    return predictedOutcome


def inferNeighbors(trainSet,testInstance,labels,k,leaveOut):   
    distances = []
    for x in range(len(trainSet)):
        distances.append((ssd.euclidean(trainSet[x], testInstance), labels[x]))
        
    distances.sort(key=operator.itemgetter(0))    
    return distances[leaveOut:k+leaveOut]

def printCIndexes(cIndex):
    print 'C Index: {a}'.format(a=cIndex)
    print
     
def calculateZScore():
    xArr = np.asarray(x)
    zScores = (xArr - xArr.mean()) / xArr.std() 
    return zScores