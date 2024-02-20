import numpy as np

inputs = [
    [1.0, -6.5, 7.22, 8.8],
    [3.14, 1.59, 2.65, -3.57],
    [0.99, -0.87, -7.27, 2.718]]
weights = [
    [0.2, 0.8, -0.5, 0.6],
    [0.55, 0.77, -0.26, 0.5],
    [0.19, -0.54, 0.01, -0.09]]
biases = [2.0, 3.0, 0.5]
weights_2 = [
    [0.66, 0.3, -0.1],
    [0.9, -0.6, 0.54],
    [0.88, -0.11, -0.67]]
biases_2 = [-1, 2, -0.5]

layer_1_out = np.dot(inputs, np.array(weights).T) + biases
layer_2_out = np.dot(layer_1_out, np.array(weights_2).T) + biases_2

print(f'Outputs: {layer_2_out}')

