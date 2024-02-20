import copy

from ops.activation import ActivationSoftmax, ASLCC
from ops.loss import LossCatCrossEntropy

import pickle


class LayerInput:

    def __init__(self):
        self.output = None

    def forward(self, inputs, training):
        self.output = inputs


class Model:
    def __init__(self):
        self.accuracy = None
        self.trainable_layers = []
        self.input_layer = None
        self.layers = []
        self.optimizer = None
        self.loss = None

    def get_params(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_params())

        return parameters

    def set_params(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_params(*parameter_set)

    def save_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_params(), f)

    @staticmethod
    def load_params(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def save(self, path):
        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('output', None)

        for layer in model.layers:
            for prop in ['inputs, output, d_inputs, d_weights, d_biases']:
                layer.__dict__.pop(prop, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, acc=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if acc is not None:
            self.accuracy = acc

    def finalize(self):
        self.input_layer = LayerInput()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], ActivationSoftmax) and \
                isinstance(self.loss, LossCatCrossEntropy):
            self.softmax_classifier_output = ASLCC()

    def train(self, X, y, *, epochs=1, batch_size=None, log_freq=1, validation=None):

        self.accuracy.init(y)

        train_steps = 1

        X_val, y_val = None, None

        if validation is not None:
            X_val, y_val = validation
            validation_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):

            print(f'Epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                batch_X = X
                batch_y = y

                if batch_size is not None:
                    batch_X = X[step * batch_size: (step + 1) * batch_size]
                    batch_y = y[step * batch_size: (step + 1) * batch_size]

                output = self.forward(batch_X, training=True)
                data_loss, reg_loss = self.loss.calculate(output, batch_y, include_reg=True)

                loss = data_loss + reg_loss

                predictions = self.output_layer_activation.predictions(output)

                acc = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                if self.optimizer is not None:
                    self.optimizer.pre_update_params()
                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)
                    self.optimizer.post_update_params()

                if not step % log_freq or step == train_steps - 1:
                    print(f'step: {step}, ',
                          f'acc: {acc:.3f}, '
                          f'loss: {loss:.3f}('
                          f'data_loss: {data_loss:.3f}, '
                          f'reg_loss: {reg_loss:}), '
                          f'lr: {self.optimizer.current_lr}')

            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accum(include_reg=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.calculate_accum()

            print(f'training, ',
                  f'acc: {epoch_acc:.3f}, '
                  f'loss: {epoch_loss:.3f}('
                  f'data_loss: {epoch_data_loss:.3f}, '
                  f'reg_loss: {epoch_reg_loss:}), '
                  f'lr: {self.optimizer.current_lr}')

            if validation is not None:
                self.evaluate(*validation, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):

        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            batch_X = X_val
            batch_y = y_val

            if batch_size is not None:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(
                output
            )
            self.accuracy.calculate(predictions, batch_y)

        val_loss = self.loss.calculate_accum(include_reg=False)
        val_acc = self.accuracy.calculate_accum()

        print(f'VALIDATION >> acc: {val_acc:.3f}, loss: {val_loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].d_inputs = self.softmax_classifier_output.d_inputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.d_inputs)

            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.d_inputs)
