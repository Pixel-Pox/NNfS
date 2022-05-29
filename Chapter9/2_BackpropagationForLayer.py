import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1. ,1. ,1.]])

# We h ave 3 sets of weights - one set for each neuron 
# we have 4 inputs, thus 4 weights
# recall that we keep wieghts transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

print(weights)

# Sum weights related to the given input multiplied by
# the gradient related to the given neuron
dx0 = sum([weights[0][0]*dvalues[0][0], weights[0][1]*dvalues[0][1], weights[0][2]*dvalues[0][2]])
dx1 = sum([weights[1][0]*dvalues[0][0], weights[1][1]*dvalues[0][1], weights[1][2]*dvalues[0][2]])
dx2 = sum([weights[2][0]*dvalues[0][0], weights[2][1]*dvalues[0][1], weights[2][2]*dvalues[0][2]])
dx3 = sum([weights[3][0]*dvalues[0][0], weights[3][1]*dvalues[0][1], weights[3][2]*dvalues[0][2]])


# Dinputs is a gradient of the neuron function with respect to inputs
dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

# Since we are using numpy arrays we can simplify the dx0-dx3 calculations
# Since the weights array is formatted so that the rows
# contain weights related to each input (weights for all neurons for the given input), we can multiply
# them by the gradient vector directly
dx0 = sum(weights[0]*dvalues[0])
dx1 = sum(weights[1]*dvalues[0])
dx2 = sum(weights[2]*dvalues[0])
dx3 = sum(weights[3]*dvalues[0])

dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

# And in fact this is literally just a dot product
weights = weights.T # just to return to the beginning and show the dot product we know

dinputs = np.dot(dvalues[0], weights)

print(dinputs)

# Let's expand the gradients to a batch of samples including inputs weights and biases
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
inputs = np.array([[1, 2, 3, 2.5],
                    [2., 5., -1., 2],
                    [-1.5, 2.7, 3.3, -0.8]])
biases = np.array([[2, 3, 0.5]])

dweights = np.dot(inputs.T, dvalues)

dinputs = np.dot(dvalues, weights)

dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dinputs)
print(dweights)
print(dbiases)

# Example for ReLU Activation derivative

# Example layer output
z = np.array([[1, 2, -3, -4],
                [2, -7, -1, 3],
                [-1, 2, 5, -1]])
dvalues = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# ReLU activation's derivative
drelu = dvalues.copy()
drelu[z <= 0] = 0

print(drelu)