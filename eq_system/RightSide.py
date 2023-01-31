import numpy as np


class RightSide:

    def __init__(self, functions):
        self.__functions = functions

    def compute(self, t, y):
        return np.array([f(t, y) for f in self.__functions])

    def __call__(self, t, y):
        return self.compute(t, y)
