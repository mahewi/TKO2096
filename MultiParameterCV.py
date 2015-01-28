'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618

Exercise 3: Applications of data analysis

The Water_data.csv file is a multi-parameter dataset consisting of 268 samples obtained from 67 mixtures of Cadmium, Lead, and tap water.  
Three features (attributes) where measured for each samples (Mod1, Mod2, Mod3).

Tasks

Use K-Nearest Neighbor Regression to predict total metal concentration (c_total), concentration of Cadmium (Cd) and concentration of Lead (Pb), for each sample.
-  The data should be normalized using z-score.
-  Implement Leave-One-Out Cross Validation approach and calculate the C-index for each output (c-total, Cd, Pb).
-  Implement Leave-Four-Out Cross Validation and calculate the C-index for each output (c-total, Cd, Pb).  This mean to leave out as test set the 4 consecutive samples at the same time (see lecture PowerPoint presentation for better explanation).

'''

import os
import numpy as np
from scipy.stats import zscore
from sklearn import preprocessing

if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "Water_data.csv"))

X = np.genfromtxt(filepath, delimiter=',', names=True, dtype='float', usecols=range(3, 6))
y = np.genfromtxt(filepath, delimiter=',', names=True, usecols=range(0,3))


def main():
    calculateZScore()

def calculateZScore():
    print X
    std_X = (X - np.mean(X))/np.std(X)
    print std_X

main()