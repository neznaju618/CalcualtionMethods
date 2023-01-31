import numpy as np


class Region:

    # Квадратная область, задаваемая своими границами a[i] <= x[i] <= b[i] , i = 1,2,...,n.
    def __init__(self, a, b):
        self.__a = a
        self.__b = b
        self.__points = None
        self.__size = 0
        self.__measure = np.prod(b - a)

    # Генерация N точек в заданной области
    def generate_points(self, N):
        self.__size = N
        self.__points = np.zeros((N, len(self.__a)))
        for i in range(len(self.__a)):
            self.__points[:, i] = np.random.uniform(self.__a[i], self.__b[i], size=N)
        return self.__points

    def get_size(self):
        return self.__size

    def get_measure(self):
        return self.__measure
