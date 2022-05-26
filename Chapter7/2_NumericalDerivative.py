import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

x = np.arange(0, 5, 0.001)
y = f(x)

# plt.plot(x, y)
# plt.show()

p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approximatex_derivative = (y2-y1)/(x2-x1)

# tangent line will be a straight line of function y = mx + b
# we know m (approximatex_derivative), x (input) so we solve for b
# b = y - mx

b = y2 - approximatex_derivative*x2

# create a function for drawing tangent_line
def tangent_line(x):
    return approximatex_derivative*x + b

# how to plot the tangent line
to_plot = [x1-0.9, x1, x1+0.9]
plt.plot(to_plot, [tangent_line(i) for i in to_plot])
plt.plot(x, y)

plt.show()

print(f'Approximate derivative for f(x)', f'where x = {1} is {approximatex_derivative}')
