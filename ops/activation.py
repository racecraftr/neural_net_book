import numpy as np

from ops.loss import LossCatCrossEntropy


class ActivationLinear:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, d_values):
        self.d_inputs = d_values.copy()

    def predictions(self, outputs):
        return outputs


class ActivationReLU:
    def __init__(self):
        self.d_inputs = None
        self.inputs = None
        self.output = None

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class ActivationSigmoid:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.d_inputs = None

    def forward(self, inputs, traning):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, d_values):
        self.d_inputs = d_values * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class ActivationSoftmax:

    def __init__(self):
        self.d_inputs = None
        self.output = None

    def forward(self, inputs, training):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, d_values):
        self.d_inputs = np.empty_like(d_values)

        for index, (single_output, single_d_values) in \
                enumerate(zip(self.output, d_values)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            self.d_inputs[index] = np.dot(jacobian_matrix, single_d_values)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class ASLCC():
    """
    Activation Softmax for Loss Categorical Cross-entropy.
    combines Softmax Activation and Cross-entropy loss for faster backward step.
    """

    def __init__(self):
        self.d_inputs = None
        self.output = None
        self.activation = ActivationSoftmax()
        self.loss = LossCatCrossEntropy()

    # def forward(self, inputs, y_true, training):
    #     self.activation.forward(inputs, training=training)
    #     self.output = self.activation.output
    #     return self.loss.calculate(self.output, y_true)

    def backward(self, d_values, y_true):
        samples = len(d_values)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.d_inputs = d_values.copy()
        self.d_inputs[range(samples), y_true] -= 1
        self.d_inputs /= samples
