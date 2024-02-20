import time

import matplotlib.pyplot as plt
from nnfs.datasets import sine_data
from ops.activation import *
from ops.loss import *
from ops.optimizer import *

nnfs.init()

X, y = sine_data()
# y = y.reshape(-1, 1)

dense1 = LayerDense(1, 64)
activation1 = ActivationReLU()

# dropout1 = Layer_Dropout(0.1)

dense2 = LayerDense(64, 64)
activation2 = ActivationReLU()

dense3 = LayerDense(64, 1)
activation3 = ActivationLinear()

loss_function = LossMSE()
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
acc_precision = np.std(y) / 250

# optimizer = Optimizer_SDG(decay=1e-3, momentum=0.8)
# optimizer = Optimizer_RMSProp(learning_rate=0.0201, decay=1e-5, rho=0.999)
t1 = time.time()

for epoch in range(10001):
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output, y)
    reg_loss = loss_function.reg_loss(dense1) + loss_function.reg_loss(dense2) + loss_function.reg_loss(dense3)

    loss = data_loss + reg_loss
    # print(loss_activation.output[5])

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < acc_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, '
              f'accuracy: {accuracy:.3f}, '
              f'loss: {loss:.3f}('
              f'data_loss: {data_loss:.3f}, '
              f'reg_loss: {reg_loss:}), '
              f'lr: {optimizer.current_lr}')

    # backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.d_inputs)
    dense3.backward(activation3.d_inputs)
    activation2.backward(dense3.d_inputs)
    dense2.backward(activation2.d_inputs)
    activation1.backward(dense2.d_inputs)
    dense1.backward(activation1.d_inputs)

    # update stuff
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

t2 = time.time()

print('training time:', t2-t1)

# plt.plot(X, y)
# plt.plot(X, activation3.output)

X_test, y_test = sine_data()
# y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()