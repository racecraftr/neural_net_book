import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T

print(np.dot(a, b))

inputs = [
    [1.0, -6.5, 7.22, 8.8],
    [3.14, 1.59, 2.65, -3.57],
    [0.99, -0.87, -7.27, 2.718]]
weights = [
    [0.2, 0.8, -0.5, 0.6],
    [0.55, 0.77, -0.26, 0.5],
    [0.19, -0.54, 0.01, -0.09]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(f'Outputs: {outputs}')