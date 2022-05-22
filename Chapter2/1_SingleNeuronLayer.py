import numpy as np


inputs = [[1, 2, 3, 2.5],
            [2, 0.6, -1, 2.3],
            [1, 4.2, -4, -0.1]
            ]
weights = [[0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = np.dot(np.array(inputs), np.array(weights).T)+biases



print(layer_outputs)