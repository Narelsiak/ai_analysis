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
        self.input_size = input_size
        self.output_size = neurons
        self.activation, self.activation_derivative = activations[activation]    
        self.optimizer = optimizer

        if input_size is not None:
            self._init_weights()

    def _init_weights(self):
        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        self.biases = np.zeros((1, self.output_size))

    def forward(self, X):
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

        dX = np.dot(dz, self.weights.T)

        self.optimizer.update(self)

        return dX
