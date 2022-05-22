import numpy as np
import math

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



# Vales from the previous output when we
# described what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# e- mathematical ocnstant, we use E here to match
# a common coding style where constants are uppercased
E = math.e

# For each value in a vector, calculate the exponential value

exp_values = []
for i in layer_outputs:
    exp_values.append(pow(E, i))
print("exponentiated values: ")
print(exp_values)

# Now normalize values
norm_base = sum(exp_values)
norm_values = []
for i in exp_values:
    norm_values.append(i/norm_base)

print("Normalized exponentiated values:")
print(norm_values)

print(f"Sum of normalized values: {sum(norm_values):.0f}")


# Same think with numpy

exp_values = np.exp(layer_outputs)
print("exponentiated values: ")
print(exp_values)

norm_values = exp_values / np.sum(exp_values)
print("Normalized exponentiated values:")
print(norm_values)

print(f"Sum of normalized values: {sum(norm_values):.0f}")


