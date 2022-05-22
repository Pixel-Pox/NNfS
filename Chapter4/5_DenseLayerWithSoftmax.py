import numpy as np
import nnfs
from nnfs.datasets import spiral_data

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


class Activation_Softmax:
    
    # Forward pass
    def forward(self, inputs) -> None:
        # get unnormalized probabilities
        # includes subtraction of the alrgest of the inputs to tackle
        # the challenge of "exploding values"
        # A value of as low as a 1000 would cause an overflow error.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Initialize
nnfs.init()

# Create a dataset
X, y = spiral_data(samples=100, classes=3)

# Create a dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation:
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation
activation2 = Activation_Softmax()

# Make forward passess through layers and activation functions that
# will take each other's outputs as inputs
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])