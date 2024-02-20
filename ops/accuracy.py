import numpy as np


class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        self.accum_sum += np.sum(comparisons)
        self.accum_ct += len(comparisons)

        return accuracy

    def calculate_accum(self):
        accuracy = self.accum_sum / self.accum_ct
        return accuracy

    def new_pass(self):
        self.accum_sum = 0
        self.accum_ct = 0


class AccuracyRegression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
