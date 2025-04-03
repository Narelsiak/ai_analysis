import numpy as np

from mlp.activations import *

activations = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative)
}

class Dense:
    def __init__(self, neurons, input_size=None, activation='relu', optimizer=None):
        self._name = "Dense"
        self.input_size = input_size
        self.output_size = neurons
        self.activation, self.activation_derivative = activations[activation]
        self.activation_name = activation
        self.optimizer = optimizer

        if input_size is not None:
            self._init_weights()

    def _init_weights(self):
        if self.activation_name == 'relu':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        elif self.activation_name in ['sigmoid', 'tanh']:
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)
        else:
            self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.biases = np.zeros((1, self.output_size))

    def forward(self, X, training=False):
        self.input = X
        self.z = np.dot(X, self.weights) + self.biases
        
        self.output = self.activation(self.z)

        return self.output
    
    def backward(self, dA):
        if self.activation.__name__ != 'softmax':
            dz = dA * self.activation_derivative(self.z)
        else:
            dz = dA

        self.dW = np.dot(self.input.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)

        dX = np.dot(dz, self.weights.T)

        self.optimizer.update(self)

        return dX

class Dropout:
    def __init__(self, rate=0.3):
        self._name = "Dropout"
        self.rate = rate
        self.mask = None

    def forward(self, X, training=False):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.rate) / (1.0 - self.rate)
            return X * self.mask
        else:
            return X

    def backward(self, dA):
        return dA * self.mask