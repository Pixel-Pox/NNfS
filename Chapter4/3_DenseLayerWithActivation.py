import numpy as np
import nnfs
from nnfs.datasets import spiral_data

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

class Activation_ReLU:
    def forward(self, inputs) -> None:
        self.output = np.maximum(0, inputs)

# Initialize
nnfs.init()

# Create a dataset
X, y = spiral_data(samples=100, classes=3)

# Create a dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation
activation1 = Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function
# Takes in output from previous layer
activation1.forward(dense1.output)

print(activation1.output[:5])