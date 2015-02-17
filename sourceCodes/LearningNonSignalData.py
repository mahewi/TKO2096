'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618
         
'''
from sklearn.neighbors import KNeighborsClassifier
from random import randint as rand
from random import shuffle as shuf
import numpy as np
from matplotlib import pyplot as pp
from scipy.stats import kendalltau as tau
import operator

if __name__ == '__main__':
    pass

def generateRandomData(size):
    features = []
    labels = []
    
    half = size / 2
    
    for i in range(size):
        features.append([rand(1, 50)])
        if i < half:     
            labels.append(0)
        else:
            labels.append(1)
            
    shuf(labels)
    
    return features, labels

def generateLoadsOfRandomData():
    features = []
    labels = []
    
    for i in range(50):
        col = []
        if (i < 25):
            labels.append(0)
        else:
            labels.append(1)
        for _ in range(1000):
            col.append(rand(1, 50))
        features.append(col) 
    
    shuf(labels)
    
    return features, labels

def selectBestCorrelations(features, labels, i, rightWay, selectCount):
    tauVals = []
    bestFeatures = []
    features = np.array(features)
    if rightWay:
        tempFeatures = np.array(filterTestInstance(features, i))
        tempLabels = np.array(filterTestInstance(labels, i))
    else:
        tempFeatures = features
        tempLabels = labels
        
    for i in range(1000):
        tauVal, _ = tau(tempFeatures[:,i], tempLabels)
        tauVals.append((abs(tauVal), i))
    
    tauVals.sort(key = operator.itemgetter(0))
    tauVals = tauVals[::-1]
    tauVals = tauVals[:10]
    for i in range(len(tauVals)):
        bestFeatures.append(features[:,tauVals[i][1]])
    
    return np.transpose(bestFeatures), labels
        
        
def leaveOneOutWithKNN(features, labels, indexOfTest):
    featuresTemp = list(features)
    labelsTemp = list(labels)
    testInstance = features[indexOfTest]
    
    del featuresTemp[indexOfTest] 
    del labelsTemp[indexOfTest]
    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(featuresTemp, labelsTemp)
    
    return neigh.predict(testInstance)

def nonSignalCV(size):
    predictions = []
    cIndexes = []
    
    for _ in range(100):
        features, labels = generateRandomData(size)
        for j in range(len(features)):
            predictions.append(leaveOneOutWithKNN(features, labels, j))
        cIndexes.append(calculateCIndex(predictions, labels))
        predictions = []
    mean, variance = calculateCIndexMeanAndVariance(cIndexes)
    performance = inferPerformance(cIndexes) * 100
    
    print 'Random data matrix size: ' + str(size)
    print 'Mean: ' + str(mean)
    print 'Variance: ' + str(variance)
    print '%-tage of C-Indexes over 0.7: ' + str(performance) + '%' 
    print 
    print
    
    pp.hist(cIndexes, 10)
    pp.xlabel('C-Index')
    pp.ylabel('Frequency')
    pp.show()
        
def calculateCIndexMeanAndVariance(cIndexes):
    mean = np.mean(cIndexes)
    variance = np.mean((cIndexes - mean)**2)
    
    return mean, variance

def inferPerformance(cIndexes):
    count = 0.0
    for i in range(len(cIndexes)):
        if (cIndexes[i] > 0.7):
            count = count + 1.0
    
    return count / len(cIndexes)

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

def filterTestInstance(listA, i):
    listTemp = list(listA)    
    del listTemp[i]
    return listTemp

def featureSelectedCV(rightWay):
    features, labels = generateLoadsOfRandomData()

    predictions = []
    
    if not rightWay:
        bestFeatures, labels = selectBestCorrelations(features, labels, 0, rightWay, 10)
    
    for i in range(len(features)):
        if rightWay:
            bestFeatures, labels = selectBestCorrelations(features, labels, i, rightWay, 10)
        predictions.append(leaveOneOutWithKNN(bestFeatures, labels, i))
    
    if not rightWay:
        print 'C-Index (wrong way): ' + str(calculateCIndex(predictions, labels))
    else:
        print 'C-Index (right way): ' + str(calculateCIndex(predictions, labels))

def printNSDHeader():
    print 'Non-signal data learning'
    print 'Random data in range (1, 49)'
    print 'Labels are binary (0, 1)'
    print '-----------------------------'
    print
    print
    
def printFSHeader():
    print 'Mis-using feature selection'
    print 'Random data in range (1, 49)'
    print 'Labels are binary (0, 1)'
    print '-----------------------------'
    print
    print

def main():
    printNSDHeader()
    nonSignalCV(10)
    nonSignalCV(50)
    nonSignalCV(100)
    nonSignalCV(500)
    
    printFSHeader()
    featureSelectedCV(False)
    featureSelectedCV(True)
    
main()