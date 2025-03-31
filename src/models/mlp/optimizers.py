import numpy as np
class Optimizer:
    def __init__(self, learning_rate=0.01, schedule=None):
        self.learning_rate = learning_rate
        self.schedule = schedule

    def update(self, layer):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def update_lr(self, epoch):
        if self.schedule:
            self.learning_rate = self.schedule.get_lr(epoch)

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.dW
        layer.biases -= self.learning_rate * layer.db

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, schedule=None):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, layer):
        if not hasattr(layer, 'm'):
            layer.m = np.zeros_like(layer.weights)
            layer.v = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)

        self.t += 1

        layer.m = self.beta1 * layer.m + (1 - self.beta1) * layer.dW
        layer.v = self.beta2 * layer.v + (1 - self.beta2) * (layer.dW ** 2)
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.db
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.db ** 2)

        m_hat = layer.m / (1 - self.beta1 ** self.t)
        v_hat = layer.v / (1 - self.beta2 ** self.t)
        m_hat_b = layer.m_b / (1 - self.beta1 ** self.t)
        v_hat_b = layer.v_b / (1 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)