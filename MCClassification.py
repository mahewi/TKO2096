'''

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
Use the K-nearest neighbor approach for the classification problem with different number of neighbors (k=3,...,10)
    * Cross validation on the training set (10-fold)
    * Test the classifier's accuracy on the test data set and report the accuracy and F-score for the best number of neighbors (K)
    * Include the confusion matrix in your report

'''

import os
import numpy as np

if __name__ == '__main__':
    pass

# Set pointer to correct destination to access the data file. 
basepath = os.path.dirname(__file__)
trainpath = os.path.abspath(os.path.join(basepath, "KDDcup99/train.csv"))
testpath = os.path.abspath(os.path.join(basepath, "KDDcup99/test.csv"))

# Parse train data into two variables trainX and trainY. x = feature data, y = outcome
trainX = np.genfromtxt(trainpath, dtype=None, delimiter=',', usecols=range(0, 41))
trainY = np.genfromtxt(trainpath, dtype=None, delimiter=',', usecols=(41))

# Parse test data into two variables testX and testY. x = feature data, y = outcome
testX = np.genfromtxt(testpath, dtype=None, delimiter=',', usecols=range(0, 41))
testY = np.genfromtxt(testpath, dtype=None, delimiter=',', usecols=(41))

