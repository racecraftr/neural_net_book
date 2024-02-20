import time

import nnfs
from nnfs.datasets import spiral_data

from ops.activation import *
from ops.layer import Layer_Dropout
from ops.loss import *
from ops.optimizer import *

nnfs.init()

X, y = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)

dense1 = LayerDense(2, 64, weight_reg_l2=5e-4, bias_reg_l2=5e-4)

activation1 = ActivationReLU()

dropout1 = Layer_Dropout(0.1)

dense2 = LayerDense(64, 1)

activation2 = ActivationSigmoid()

loss_function = LossBinCrossEntropy()

optimizer = OptimizerAdam(learning_rate=0.03, decay=5e-7)

loss_activation = ASLCC()

# optimizer = Optimizer_SDG(decay=1e-3, momentum=0.8)
# optimizer = Optimizer_RMSProp(learning_rate=0.0201, decay=1e-5, rho=0.999)
t1 = time.time()

for epoch in range(10001):
    # forward pass
    dense1.forward(X, training=True)
    activation1.forward(dense1.output, training=True)
    # dropout1.forward(activation1.output)
    dense2.forward(activation1.output, training=True)
    activation2.forward(dense2.output, traning=True)

    data_loss = loss_function.calculate(activation2.output, y)
    reg_loss = loss_function.reg_loss() + loss_function.reg_loss()

    loss = data_loss
    # print(loss_activation.output[5])

    predictions = (activation2.output > 0.5) * 1
    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, '
              f'accuracy: {accuracy:.3f}, '
              f'loss: {loss:.3f}, '
              f'data_loss: {data_loss:.3f}, '
              f'reg_loss: {reg_loss:}, '
              f'lr: {optimizer.current_lr}')

    # backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.d_inputs)
    dense2.backward(activation2.d_inputs)
    activation1.backward(dense2.d_inputs)
    dense1.backward(activation1.d_inputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    # update stuff
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

t2 = time.time()

print('training time:', t2-t1)

X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

dense1.forward(X_test, training=False)
activation1.forward(dense1.output, training=False)
dense2.forward(activation1.output, training=False)
loss = loss_function.calculate(dense2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation accuracy: {accuracy:.3f}, loss: {loss:.3f}')