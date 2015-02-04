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
import matplotlib.pyplot as pp


if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
inputpath = os.path.abspath(os.path.join(basepath, "Data4/INPUT.csv"))
outputpath = os.path.abspath(os.path.join(basepath, "Data4/OUTPUT.csv"))
coordinatespath = os.path.abspath(os.path.join(basepath, "Data4/COORDINATES.csv"))

x = np.genfromtxt(inputpath, delimiter=',')
y = np.genfromtxt(outputpath, delimiter=',')
z = np.genfromtxt(coordinatespath, delimiter=',')

xArr = np.asarray(x)
stdX = (xArr - xArr.mean()) / xArr.std() 


def calculateDistanceMatrix():
    distanceMatrix = []
    for i in range(len(z)):
        xAxis = []
        for j in range(len(z)):
            if i == j:
                xAxis.append(-1.0)
            else:
                xAxis.append(ssd.euclidean(z[i], z[j]))
        distanceMatrix.append(xAxis)
   
    return distanceMatrix

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

def LooCV(k, distanceMatrix):
    yPredictions = []
    for i in range(len(stdX)):
        neighbors = inferNeighbors(stdX, stdX[i], y, k, distanceMatrix[i])
        yPredictions.append(chooseMajorityLabel(neighbors,k))
        
    cIndex = calculateCIndex(yPredictions, y)
    printCIndexes(cIndex)
    
    return cIndex

def chooseMajorityLabel(neighbors, k):
    predictedOutcome = []
    sumOfMod = 0.0
    for i in range(len(neighbors)):
        sumOfMod = sumOfMod + neighbors[i][1]
    predictedOutcome.append(sumOfMod/k)
    
    return predictedOutcome


def inferNeighbors(trainSet, testInstance, labels, k, distRow):   
    distances = []
    for x in range(len(trainSet)):
        if distRow[x] >= 0.0:
            distances.append((ssd.euclidean(trainSet[x], testInstance), labels[x]))
        
    distances.sort(key=operator.itemgetter(0))  
    return distances[0:k]

def printCIndexes(cIndex):
    print 'C-Index: {a}'.format(a=cIndex)
    print

def calculateDeadZone(matrix):
    for i in range(len(matrix)):
        xAxis = matrix[i]
        for _ in range(10):
            minIndex = xAxis.index(min(filter(lambda x:x>=0.0, xAxis)))
            xAxis[minIndex] = -1.0
    
    return matrix

def plotCIndexVsDeadZone(cIndexes, deadZoneValues):
    pp.ylabel('C-index')
    pp.xlabel('Deadzone radius')
    pp.plot(deadZoneValues, cIndexes)
    pp.show()     
     
def main():
    distanceMatrix = calculateDistanceMatrix()
    cIndexes = []
    deadZoneValues = []
    for i in range(5):
        print 'Leave-one-out CV with deadzone radius ' + str(i * 10) + ':'
        cIndexes.append(LooCV(5, distanceMatrix))
        deadZoneValues.append(i * 10)
        distanceMatrix = calculateDeadZone(distanceMatrix)
    
    plotCIndexVsDeadZone(cIndexes, deadZoneValues)

main()