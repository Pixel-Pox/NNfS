import matplotlib.pyplot as plt
import numpy as np

def f(x) -> float:
    return 2*x**2

def approximate_tangent_line(x, approximate_derivative) -> float:
    return approximate_derivative*x + b

def tangent_line(x) -> float:
    return (approximatex_derivative*x) + b

x = np.arange(0, 5, 0.001)
y = f(x)

colors = ['k', 'g', 'r', 'b', 'c']


for i in range(len(colors)):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    approximatex_derivative = (y2-y1)/(x2-x1)

    b = y2 - (approximatex_derivative*x2)

    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot(x,y)
    plt.plot([point for point in to_plot],
            [approximate_tangent_line(point, approximatex_derivative) for point in to_plot], c=colors[i])

    print(f'Approximate derivative for f(x)', f'where x = {x1} is {approximatex_derivative}')

plt.show()


