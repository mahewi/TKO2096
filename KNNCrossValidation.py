'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618

Exercise 1: Application of data analysis

K-nearest neighbors and cross-validation
- Program your own implementation of kNN. Preferably with python.
- Select a data set of your choice from the UCI machine learning repository: \url{http://archive.ics.uci.edu/ml/}
- Use cross-validation on the training set and report the performance (e.g. classification or regression error) for different values of $k$

Phases: 
1. Parse data: Divide data into test and train sets.
2. Similarity: Calculate the distances data instances.
3. Neighbors: Locate k most similar data instances.
4. Response: Generate a response from a set of data instances.
5. Accuracy: Summarize the accuracy of predictions.

'''


import os
import operator
import numpy as np
import scipy.spatial.distance as ssd

if __name__ == '__main__':
    pass

# Set pointer to correct destination. fertility_diag.txt contains: 
#     - 100 volunteers provide a semen sample analyzed according to the WHO 2010 criteria. 
#       Sperm concentration are related to socio-demographic data, environmental factors, health status, and life habits.
#     - Number of instances : 100
#     - Number of attributes : 10


basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "fertility_diag.txt"))

# Parse data into two variables x and y. x = feature data, y = outcome
x = np.loadtxt(filepath, delimiter=',', usecols=range(0,9))
y = np.genfromtxt(filepath, dtype='str', delimiter=',', usecols=(9))

# Split the data into four sets: 
#    - Train/test sets for feature data and train/test sets for outcome labels.
def splitToTrainAndTest():
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []
    
    for i in range(len(x) - 1):
        if (i < (len(x)*0.50)):
            yTest.append(y[i])
            xTest.append(x[i])
        else:
            yTrain.append(y[i])
            xTrain.append(x[i])
            
    return xTrain, yTrain, xTest, yTest


# Calculate eucledian distance against a single test instance against each train instance.
def calculateDistances(trainSet, testInstance, trainLabel):
    distances = []
    
    for x in range(len(trainSet)):
        distances.append((ssd.euclidean(trainSet[x], testInstance), trainLabel[x]))
        
    distances.sort(key=operator.itemgetter(0))
    
    return distances


def inferNeighbours(distances, k):
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x])
        
    return neighbours
        

def chooseMajorityLabel(neighbours):
    labelCounts = {}
    
    for x in range(len(neighbours)):
        label = neighbours[x][1]
        if label in labelCounts:
            labelCounts[label] += 1
        else:
            labelCounts[label] = 1
    return sorted(labelCounts.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]


def calculateErrorPercentage(predictedLabels, yTest):
    errorCount = 0.0
    for x in range(len(predictedLabels)):
        if predictedLabels[x] != yTest[x]:
            errorCount += 1.0
    
    return errorCount/len(yTest)*100.0
            

def printOutcome(errorPercentage, k):
    print 'The accuracy of the predition with value k={k}: {errorPercentage} %'.format(k=k, errorPercentage=errorPercentage)

    
# Run method
def mainMethod(k):
    xTrain, yTrain, xTest, yTest = splitToTrainAndTest()
    eucDistances = []
    neighbours = []
    predictedLabels = []
    
    for y in range(len(xTest)):
        eucDistances = calculateDistances(xTrain, xTest[y], yTrain)
        neighbours = inferNeighbours(eucDistances, k)
        predictedLabels.append(chooseMajorityLabel(neighbours))
    
    errorPercentage = calculateErrorPercentage(predictedLabels, yTest)
    
    printOutcome(errorPercentage, k)
    
    
for i in range(1,31):    
    mainMethod(i)   