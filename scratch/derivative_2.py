import numpy as np

z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
d_values = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

d_relu = d_values.copy()
d_relu[z <= 0] = 0
print(d_relu)