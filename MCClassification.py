'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618

Exercise 2: Applications of data analysis

In this exercise the KDDcup99 intrusion detection dataset is given. The actual KDDcup training data set contain 38 different network attack types.

KDDcup99 Dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

The data given for this exercise is a simplified version of KDDcup99 containing 5 categories(Classes) of traffic.
Normal class = 1 
DoS = 2
Probe = 3
R2L = 4
U2R = 5

A training dataset of 5600 different labeled (Last column) network connection data instances for constructing the classifier.
A test data set of 3100 datapoint (different distribution from training data) is given for testing the classifier, data is labeled   (Last column). 

Task:
Use the K-nearest neighbor approach for the classification problem with different number of neighbors (k = 3,..., 10)
    * Cross validation on the training set (10-fold)
    * Test the classifier's accuracy on the test data set and report the accuracy and F-score for the best number of neighbors (K)
    * Include the confusion matrix in your report

'''

import os
import numpy as np
import scipy.spatial.distance as ssd
import operator
from matplotlib import pyplot as pp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as fscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import StratifiedKFold

if __name__ == '__main__':
    pass

# Set pointer to correct destination to access the data file. 
basepath = os.path.dirname(__file__)
trainpath = os.path.abspath(os.path.join(basepath, "KDDcup99/train.csv"))
testpath = os.path.abspath(os.path.join(basepath, "KDDcup99/test.csv"))

# Parse train data into two variables trainX and trainY. x = feature data, y = outcome
trainX = np.genfromtxt(trainpath, delimiter=',', usecols=range(0, 41))
trainY = np.genfromtxt(trainpath, dtype=None, delimiter=',', usecols=(41))

# Parse test data into two variables testX and testY. x = feature data, y = outcome
testX = np.genfromtxt(testpath, delimiter=',', usecols=range(0, 41))
testY = np.genfromtxt(testpath, dtype=None, delimiter=',', usecols=(41))

dataLimit = 1000

def main():
    preProcessedTrainX = preProcessData(trainX[:dataLimit],trainpath)
    preProcessedTestX = preProcessData(testX[:dataLimit],testpath)
    normalizedTrainX = normalize(preProcessedTrainX)
    normalizedTestX = normalize(preProcessedTestX)
    
    xTrain, xTest, yTrain, yTest = crossValidate(normalizedTrainX)
    
    K = selectBestK(xTrain,xTest,yTrain,yTest)
    
    calculateAccuracyAndScore(K,normalizedTrainX,normalizedTestX)
    
def calculateAccuracyAndScore(K,normTrainX,normTestX):
    knc = KNeighborsClassifier(n_neighbors=K)
    knc.fit(normTrainX, trainY[:dataLimit])
    predictedLabels = knc.predict(normTestX)
    print calculateErrorPercentage(predictedLabels, testY[:dataLimit])
    print fscore(testY[:dataLimit],predictedLabels)
    cm = confusion_matrix(testY[:dataLimit], predictedLabels)
    
    pp.matshow(cm)
    pp.show()

def selectBestK(xTrain,xTest,yTrain,yTest):
    K = 3
    errorOfBestK = 1.0
    for i in range(3,11):
        eucDistances = []
        neighbors = []
        predictedLabels = []
        
        for j in range(len(xTest)):
            eucDistances = calculateDistances(xTrain, xTest[j], yTrain)
            neighbors = inferNeighbors(eucDistances, i)
            predictedLabels.append(chooseMajorityLabel(neighbors))
        
        errorPercentage = calculateErrorPercentage(predictedLabels, yTest)
        if(errorPercentage < errorOfBestK):
            K = i
    return K
    
def crossValidate(normX):
    skf = StratifiedKFold(trainY[:dataLimit], n_folds=10, shuffle=True)
    for train_index, test_index in skf:
        X_train, X_test = normX[train_index], normX[test_index]
        y_train, y_test = trainY[train_index], trainY[test_index]
    
    return X_train, X_test, y_train, y_test

def preProcessData(x,path):
    stringColumns = np.genfromtxt(path, delimiter=',',dtype=None, usecols=range(1,4))
    transportProtocols = []
    transferProtocols = []
    statusCols = []
    for i in range(len(stringColumns)):
        transportProtocols.append(stringColumns[i][0])
        transferProtocols.append(stringColumns[i][1])
        statusCols.append(stringColumns[i][2])
        
    labelE = LabelEncoder()
    labelE.fit(transportProtocols)
    encodedSecond = labelE.transform(transportProtocols)
    labelE.fit(transferProtocols)
    encodedThird = labelE.transform(transferProtocols)
    labelE.fit(statusCols)
    encodedForth = labelE.transform(statusCols)
    
    enc = OneHotEncoder()
    encodedMatrix = []
    for i in range(len(encodedSecond)):
        temp = []
        temp.append(encodedSecond[i])
        temp.append(encodedThird[i])
        temp.append(encodedForth[i])
        encodedMatrix.append(temp)
        
    enc.fit(encodedMatrix)
    encodedStrings = enc.transform(encodedMatrix).toarray()
    
    for i in range(len(x)):
        temp = encodedStrings[i]
        x[i][1] = np.where(encodedStrings[i][0:enc.n_values_[0]]==1)[0]
        x[i][2] = np.where(encodedStrings[i][enc.n_values_[0]:enc.n_values_[1]+enc.n_values_[0]]==1)[0]
        x[i][3] = np.where(encodedStrings[i][enc.n_values_[1]+enc.n_values_[0]:enc.n_values_[2]+enc.n_values_[1]+enc.n_values_[0]]==1)[0]
        
    return x







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

    

main()
