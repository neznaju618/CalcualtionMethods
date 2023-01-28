from enum import Enum

import numpy as np


class ErrorFunctions(Enum):
    MSE = (lambda y_pred, y_true: np.sqr(y_pred - y_true) / 2, lambda y_pred, y_true: y_pred - y_true)
    LOG = (lambda y_pred, y_true: -y_true.dot(np.log(y_pred)), lambda y_pred, y_true: -y_true/y_pred)

    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
