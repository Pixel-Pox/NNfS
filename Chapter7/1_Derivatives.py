import numpy as np
import matplotlib.pyplot as plt

# define a non-linear function
def f(x):
    return 2*x**2

# create a sample array of points - coordinates x and y
x = np.array(range(5))
y = f(x)

# print the results
print(x)
print(y)

# to calculate measure of impact x has on y
# we will calculate difference between two points
# Delta X / Delta Y
print((y[1] - y[0])/([x[1] - x[0]]))
print((y[2] - y[1])/([x[2] - x[1]]))

# This however prints different values as the
# function grows exponentialy

# if we however take points exteremely close to
# each other then we'll see the approximation of
# derivative for that function (which is equal to 4x)

p2_delta = 0.0001
x1 = 1
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approximatex_derivative = (y2-y1)/(x2-x1)
print(approximatex_derivative)