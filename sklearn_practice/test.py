# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib# Generate a dataset and plot it
# np.random.seed(0)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


from pylab import *

import numpy as np
arrays = [np.random.randn(3, 4) for _ in range(10)]

print np.stack(arrays, axis=0).shape
