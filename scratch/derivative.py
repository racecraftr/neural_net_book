import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2 * x**2 * np.sin(x)


x = np.arange(0, 5, 0.001)
y = f(x)
plt.plot(x, y)

p2_delta = 0.0001
colors = 'kgrbc'

for i in range(5):
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    approx_deriv = (y2 - y1) / (x2 - x1)
    b = y2 - approx_deriv * x2


    def tangent_line(x):
        return approx_deriv*x + b

    c = colors[i % len(colors)]
    to_plot = [x1 - 0.9, x1, x1 + 0.9]
    plt.scatter(x1, y1, c=c)
    plt.plot(to_plot, [tangent_line(i) for i in to_plot], c=c)

plt.show()



