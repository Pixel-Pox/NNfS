# Forward pass
x = [1.0, -2.0, 3.0] # input
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
#print(xw0, xw1, xw2)

# Adding weighed inputs and a bias
z = xw0 + xw1 + xw2 + b
#print(z)

# ReLU activation function
y = max(0, z)
#print(y)

# Backward pass

# The derivative from the next layer, 1 for simplification
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue*(1. if z > 0 else 0.)
print(drelu_dz)


# partial derivative of a sum is always 1
# partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db

print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
# Derivative of weight*input will be either weight or input
# base on in relation to what we are doing the calculation
dmul_dx0 = w[0]
drelu_dx0 = drelu_dxw0 * dmul_dx0
dmul_dx1 = w[1]
drelu_dx1 = drelu_dxw1 * dmul_dx1
dmul_dx2 = w[2]
drelu_dx2 = drelu_dxw2 * dmul_dx2

dmul_dw0 = x[0]
drelu_dw0 = drelu_dxw0 * dmul_dw0
dmul_dw1 = x[1]
drelu_dw1 = drelu_dxw1 * dmul_dw1
dmul_dw2 = x[2]
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# Or to simplify: 
drelu_dx0 = drelu_dxw0 * w[0]
drelu_dx1 = drelu_dxw1 * w[1]
drelu_dx2 = drelu_dxw2 * w[2]

drelu_dw0 = drelu_dxw0 * x[0]
drelu_dw1 = drelu_dxw1 * x[1]
drelu_dw2 = drelu_dxw2 * x[2]

# Where if dsum_dxw0 = 1 then and since drelu_dxw0 = drelu_dz * dsum_dxw0 then drelu_dxw0 = drelu_dz:

drelu_dx0 = drelu_dz * w[0] 
drelu_dx1 = drelu_dz * w[1]
drelu_dx2 = drelu_dz * w[2]

drelu_dw0 = drelu_dz * x[0] 
drelu_dw1 = drelu_dz * x[1]
drelu_dw2 = drelu_dz * x[2]

# and since drelu_dz = dvalue * (1. if z > 0 else 0.) then we just substitute
drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0] 
drelu_dx1 = dvalue * (1. if z > 0 else 0.) * w[1]
drelu_dx2 = dvalue * (1. if z > 0 else 0.) * w[2]

drelu_dw0 = dvalue * (1. if z > 0 else 0.) * x[0] 
drelu_dw1 = dvalue * (1. if z > 0 else 0.) * x[1]
drelu_dw2 = dvalue * (1. if z > 0 else 0.) * x[2]

# All together the partial derivatives above, combined into a vector 
# Make up aour gradients. Our gradients could be represented as:
dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db

print(w, b)

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w, b)

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding
z = xw0 + xw1 + xw2 + b
# ReLU activation function
y = max(z, 0)
print(y)

# We’ve successfully decreased this neuron’s output from 6.000 to 5.985. Note that it does not
# make sense to decrease the neuron’s output in a real neural network; we were doing this purely
# as a simpler exercise than the full network.