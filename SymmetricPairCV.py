'''

Authors: Marco Willgren, 502606
         Jarno Vuorenmaa, 503618
         
'''

import os
import numpy as np

if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
featurepath = os.path.abspath(os.path.join(basepath, "Data5/proteins.features"))
labelpath = os.path.abspath(os.path.join(basepath, "Data5/proteins.labels"))

x = np.genfromtxt(featurepath, delimiter=',')
y = np.genfromtxt(labelpath, delimiter=',')