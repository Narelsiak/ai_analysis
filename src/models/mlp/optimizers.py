class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError("This method should be overridden by subclasses")
    
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.dW
        layer.biases -= self.learning_rate * layer.db

class Adam(Optimizer):
    def __init__(self):
        super().__init__()
        pass