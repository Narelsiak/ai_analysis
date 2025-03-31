import numpy as np

class ExponentialDecay:
    def __init__(self, initial_lr, decay_rate):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, epoch):
        return self.initial_lr * np.exp(-self.decay_rate * epoch)

class StepDecay:
    def __init__(self, initial_lr, drop_factor, step_size):
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.step_size = step_size

    def get_lr(self, epoch):
        return self.initial_lr * (self.drop_factor ** (epoch // self.step_size))