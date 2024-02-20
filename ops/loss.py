import numpy as np


class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def reg_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_reg_l1 > 0:
                regularization_loss += layer.weight_reg_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_reg_l2 > 0:
                regularization_loss += layer.weight_reg_l2 * np.sum(np.square(layer.weights))

            if layer.bias_reg_l1 > 0:
                regularization_loss += layer.bias_reg_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_reg_l2 > 0:
                regularization_loss += layer.bias_reg_l2 * np.sum(np.square(layer.biases))

        return regularization_loss

    def calculate(self, output, y, *, include_reg=False):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accum_sum += np.sum(sample_losses)
        self.accum_ct += len(sample_losses)

        if not include_reg:
            return data_loss
        return data_loss, self.reg_loss()

    def calculate_accum(self, *, include_reg=False):

        data_loss = self.accum_sum / self.accum_ct

        if not include_reg:
            return data_loss
        return data_loss, self.reg_loss()

    def new_pass(self):
        self.accum_sum = 0
        self.accum_ct = 0

    def forward(self, output, y):
        pass


class LossCatCrossEntropy(Loss):
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        correct_confidences = np.array([])
        samples = len(y_pred)

        # do some clipping to prevent zero values.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, axis=1
            )

        if len(correct_confidences) == 0:
            raise ValueError(f"y_true has {len(y_true.shape)} dimension(s), can only be either 1 or 2")

        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods

    def backward(self, d_values, y_true):
        samples = len(d_values)
        labels = len(d_values[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            self.d_inputs = -y_true / (d_values + 1e-8)
            self.d_inputs = self.d_inputs / samples


class LossBinCrossEntropy(Loss):
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])

        clipped_d_values = np.clip(d_values, 1e-7, 1 - 1e-7)

        self.d_inputs = -(y_true / clipped_d_values -
                          (1 - y_true) / (1 - clipped_d_values)) / outputs

        self.d_inputs = self.d_inputs / samples


class LossMSE(Loss):
    """
    mean squared error loss.
    """
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])
        self.d_inputs = -2 * (y_true - d_values) / outputs
        self.d_inputs = self.d_inputs / samples


class LossMAE(Loss):
    """
    mean absolute error loss.
    """

    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, d_values, y_true):
        samples = len(d_values)
        outputs = len(d_values[0])

        self.d_inputs = np.sign(y_true - d_values) / outputs
        self.d_inputs = self.d_inputs / samples
