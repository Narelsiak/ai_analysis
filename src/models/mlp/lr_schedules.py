import numpy as np

class ExponentialDecay:
    def __init__(self, initial_lr, decay_rate, min_lr=0):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    def get_lr(self, epoch):
        lr = self.initial_lr * np.exp(-self.decay_rate * epoch)
        return max(self.min_lr, lr)

class StepDecay:
    def __init__(self, initial_lr, drop_factor, step_size, min_lr=0):
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.step_size = step_size
        self.min_lr = min_lr

    def get_lr(self, epoch):
        lr = self.initial_lr * (self.drop_factor ** (epoch // self.step_size))
        return max(self.min_lr, lr)