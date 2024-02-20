import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons, weight_reg_l1=0, weight_reg_l2=0, bias_reg_l1=0, bias_reg_l2=0):
        super().__init__()
        self.d_inputs = None
        self.d_biases = None
        self.d_weights = None
        self.output = None
        self.inputs = None

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_reg_l1 = weight_reg_l1
        self.weight_reg_l2 = weight_reg_l2

        self.bias_reg_l1 = bias_reg_l1
        self.bias_reg_l2 = bias_reg_l2

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values, y_true=None):
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        if self.weight_reg_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.d_weights += self.weight_reg_l1 * dL1

        if self.weight_reg_l2 > 0:
            self.d_weights += 2 * self.weight_reg_l2 * self.weights

        if self.bias_reg_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.d_biases += self.bias_reg_l1 * dL1

        if self.bias_reg_l2 > 0:
            self.d_biases += 2 * self.bias_reg_l2 * self.biases

        self.d_inputs = np.dot(d_values, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
        self.inputs = None
        self.binary_mask = None
        self.d_inputs = None
        self.output = None

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(
            1, self.rate, size=inputs.shape
        ) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, d_values, y_true=None):
        self.d_inputs = d_values * self.binary_mask
