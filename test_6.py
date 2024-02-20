from data import loading
from model.model import Model
from ops.accuracy import AccuracyCategorical
from ops.activation import *
from ops.layer import LayerDense
from ops.loss import LossCatCrossEntropy
from ops.optimizer import OptimizerAdam

# to get mnist data, download zip file here: 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# you may need to use p7zip (7za.exe) to unzip it.
X, y, X_test, y_test = loading.create_data_mnist('fashion_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys) # limbo reference? lmao
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 10))
model.add(ActivationSoftmax())

model.set(
    loss=LossCatCrossEntropy(),
    optimizer=OptimizerAdam(decay=5e-5),  # make sure that the learning rate is low by default or is set to be low.
    acc=AccuracyCategorical()
)

model.finalize()
model.train(X, y, validation=(X_test, y_test), epochs=5, batch_size=128, log_freq=100)

params = model.get_params()

model = Model()
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())

model.set(
    loss=LossCatCrossEntropy(),
    acc=AccuracyCategorical()
)

model.finalize()
model.set_params(params)
model.evaluate(X_test, y_test)
