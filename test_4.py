# from nnfs.datasets import sine_data, spiral_data
import nnfs
from nnfs.datasets import spiral_data

from ops.activation import *
from ops.layer import Layer_Dropout
from ops.loss import LossCatCrossEntropy
from model.model import Model
from ops.optimizer import *
from ops.accuracy import AccuracyCategorical

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(LayerDense(2, 512, weight_reg_l2=5e-4, bias_reg_l2=5e-4))
model.add(ActivationReLU())
model.add(Layer_Dropout(0.1))
model.add(LayerDense(512, 3))
model.add(ActivationSoftmax())

model.set(
    loss=LossCatCrossEntropy(),
    optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-5),
    acc=AccuracyCategorical()
)

model.finalize()

model.train(X, y, validation=(X_test, y_test), epochs=10001, log_freq=100)
