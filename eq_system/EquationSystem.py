import numpy as np


class EquationSystem:

    def __init__(self, right_side, start_condition, a, b):
        self.__F = right_side
        self.__start_cond = start_condition
        self.__a = a
        self.__b = b

    def solve(self, N):
        h = (self.__b - self.__a) / (N - 1)
        result = np.zeros((len(self.__start_cond), N))
        result[:, 0] = self.__start_cond
        for n in range(1, N):
            t = self.__a + h*n
            forecast = result[:, n-1] + h*self.__F(t, result[:, n-1])
            result[:, n] = result[:, n-1] + h * (self.__F(t - h, result[:, n-1]) + self.__F(t, forecast)) / 2
        return result
