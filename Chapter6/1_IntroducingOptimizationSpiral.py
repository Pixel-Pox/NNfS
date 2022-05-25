import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

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

# Common loss class
class Loss:

    # Calculates the data and regylarization losses
    # giben model output and ground truth values
    def calculate(self, output, y) -> float:

        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss

    
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    
    # forward pass
    def forward(self, y_pred, y_true) -> list:

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only fo rone-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

nnfs.init()

# Create a dataset
X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

# Create a model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Update weights and biases by a marginal random value
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    # Perform a forward pass of the training data through first layer
    dense1.forward(X)
    # Activation function with first layers output
    activation1.forward(dense1.output)
    # Forward pass for second layer with output of the activation function
    dense2.forward(activation1.output)
    # Model's output
    activation2.forward(dense2.output)

    # Calculate loss of the model
    loss = loss_function.calculate(activation2.output, y)

    # predictions and accuracy of the current model
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # if new solution produces less loss then the previous 
    # then mark these weights and biases as new best variables
    if loss < lowest_loss:
        print(f'New set of weights found, iteration: {iteration}\nloss: {loss}\nacc: {accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    # if not then revert to the previous weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

