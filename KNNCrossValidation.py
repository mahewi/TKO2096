'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618

Exercise 1: Application of data analysis

K-nearest neighbors and cross-validation
- Program your own implementation of kNN. Preferably with python.
- Select a data set of your choice from the UCI machine learning repository: \url{http://archive.ics.uci.edu/ml/}
- Use cross-validation on the training set and report the performance (e.g. classification or regression error) for different values of $k$

Phases: 
1. Divide data into test and train sets.
2. Calculate the distances data instances.
3. Locate k most similar data instances.
4. Generate a response from a set of data instances.
5. Summarize the accuracy of the predictions.

'''


import os
import random
import operator
import numpy as np
import scipy.spatial.distance as ssd

if __name__ == '__main__':
    pass

# fertility_diag.txt contains: 
#     - 100 volunteers provide a semen sample analyzed according to the WHO 2010 criteria. 
#       Sperm concentration are related to socio-demographic data, environmental factors, health status, and life habits.
#     - Number of instances : 100
#     - Number of attributes : 10

# Set pointer to correct destination to access the data file. 
basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "Data1/fertility_diag.txt"))

# Parse data into two variables x and y. x = feature data, y = outcome
x = np.loadtxt(filepath, delimiter=',', usecols=range(0,9))
y = np.genfromtxt(filepath, dtype='str', delimiter=',', usecols=(9))


# Split the data into four sets: 
#    - Train/test sets for feature data and train/test sets for outcome labels (O/N). 
#    - Method argument == posibility to be included into test set
#    - Returns train and test sets
def splitToTrainAndTest(split):
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []
    
    for i in range(len(x) - 1):
        if (random.random() < split):
            yTest.append(y[i])
            xTest.append(x[i])
        else:
            yTrain.append(y[i])
            xTrain.append(x[i])
            
    return xTrain, yTrain, xTest, yTest


# Calculate eucledian distances for a single test instance against each train instance.
# Return list of distances with label in increasing order
def calculateDistances(trainSet, testInstance, trainLabel):
    distances = []
    
    for x in range(len(trainSet)):
        distances.append((ssd.euclidean(trainSet[x], testInstance), trainLabel[x]))
        
    distances.sort(key=operator.itemgetter(0))
    
    return distances


# Returns k nearest neighbors (sorted list)
def inferNeighbors(distances, k):
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
        
    return neighbors
        

# Calculates labels of the neighbors
# Returns the most popular neighbor
def chooseMajorityLabel(neighbors):
    labelCounts = {}
    
    for x in range(len(neighbors)):
        label = neighbors[x][1]
        if label in labelCounts:
            labelCounts[label] += 1
        else:
            labelCounts[label] = 1
    return sorted(labelCounts.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]


# Compares the predicted labels to corresponding labels in the test set 
# Returns the error percentage of the comparison
def calculateErrorPercentage(predictedLabels, yTest):
    errorCount = 0.0
    for x in range(len(predictedLabels)):
        if predictedLabels[x] != yTest[x]:
            errorCount += 1.0
    
    return errorCount/len(yTest)*100.0
            

# Print the resulting accuracy of the prediction with value K
def printOutcome(errorPercentage, k):
    print 'The accuracy of the predition with value k={k}: {errorPercentage} %'.format(k=k, errorPercentage=errorPercentage)

    
# Run method
# Calculate KNN with different values of K.
def mainMethod():
    xTrain, yTrain, xTest, yTest = splitToTrainAndTest(0.40)
    k = 31
    
    for i in range(1, k):
        eucDistances = []
        neighbors = []
        predictedLabels = []
        
        for y in range(len(xTest)):
            eucDistances = calculateDistances(xTrain, xTest[y], yTrain)
            neighbors = inferNeighbors(eucDistances, i)
            predictedLabels.append(chooseMajorityLabel(neighbors))
        
        errorPercentage = calculateErrorPercentage(predictedLabels, yTest)
        
        printOutcome(errorPercentage, i)
        
    
      
mainMethod()   