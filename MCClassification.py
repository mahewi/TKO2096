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

if __name__ == '__main__':
    pass
