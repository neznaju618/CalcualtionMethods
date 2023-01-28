from enum import Enum

import numpy as np


class ActivationFunctions(Enum):
    LINEAR = (lambda x: x, lambda x: 1 + x * 0)
    ReLU = (lambda x: (np.abs(x) + x) / 2, lambda x: np.heaviside(x, 0))
    SIGMOID = (lambda x: 1 / (1 + np.exp(-x)), lambda x: 1 - 1 / (1 + np.exp(-x)))
    TANH = (lambda x: np.tanh(x), lambda x: 1 - np.square(np.tanh(x)))
    SOFTMAX = (
        lambda y: np.exp(-y) / np.sum(np.exp(-y)),
        lambda y: (-np.exp(-y) * np.sum(np.exp(-y)) + np.square(np.exp(-y))) / np.square(np.sum(np.exp(-y))))

    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
