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

print x
print y
print z