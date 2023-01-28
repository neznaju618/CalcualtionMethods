import numpy as np


class DataPreprocessor:

    def __init__(self, M):
        self.__deg = M + 1

    def preprocess(self, x):
        data = np.ones((len(x), self.__deg))
        for i in range(1, self.__deg):
            data[:, i] = data[:, i - 1] * x
        return data

    def z_normalize(self, data):
        for i in range(len(data)):
            std = np.std(data[i])
            mean = np.mean(data[i])
            data[i] -= mean
            data[i] /= std
        return data

    def min_max_normalize(self, data):
        for i in range(len(data)):
            min_x = np.min(data[i])
            max_x = np.max(data[i])
            data[i] -= min_x
            data[i] /= (max_x - min_x)
        return data
