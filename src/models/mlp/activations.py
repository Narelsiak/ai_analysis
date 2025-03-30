import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    """Derivative of Sigmoid activation function."""
    return x * (1 - x)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of Tanh activation function."""
    return 1 - np.tanh(x)**2

def softmax(x):
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_derivative(x):  
    """Derivative of Softmax activation function."""
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)