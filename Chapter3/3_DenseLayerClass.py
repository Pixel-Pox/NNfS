import numpy as np
from nnfs.datasets import spiral_data
import nnfs

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons) -> None:
        # Initialize wieghts and biases

        # Random weights for start with the shape of (inputs, neurons) which ensures 
        # that we don't have to transpose it during forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # biases are set to zero to make sure all of them fire initially, sometimes 
        # it might be necessary to select values other then 0 
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs) -> None:
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])