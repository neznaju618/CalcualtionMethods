import numpy as np


class MNKDataset:

    def __init__(self, a, b, f):
        self.__a = a
        self.__b = b
        self.__f = f
        self.__data_x = None
        self.__data_y = None

    def initialize(self, N):
        self.__data_x = self.__a + np.random.random(size=N) * (self.__b-self.__a)
        self.__data_y = self.__f(self.__data_x) + np.random.normal(0, 0.5, size=N)

    def generate_test_dataset(self, N):
        res = MNKDataset(self.__a, self.__b, self.__f)
        res.initialize(N)
        return res

    def x(self):
        return self.__data_x

    def y(self):
        return self.__data_y

    def get_subset(self, start_idx, finish_idx):
        res = MNKDataset(self.__a, self.__b, self.__f)
        res.__data_y = self.__data_y[start_idx:finish_idx].copy()
        res.__data_x = self.__data_x[start_idx:finish_idx].copy()
        return res

    def __len__(self):
        return len(self.__data_x)
