import numpy as np
import matplotlib.pyplot as plt
import nnfs 
from nnfs.datasets import spiral_data
# sets the radon seed to 0, creates float32 dtype default, and override the original dot product from NumPy
nnfs.init()

X, y, = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()