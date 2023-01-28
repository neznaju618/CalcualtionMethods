import numpy as np


class Metric:

    def d_1(o1, o2):
        return np.sum(np.abs(o1 - o2))

    def d_2(o1, o2):
        return np.sum(np.square(o1 - o2))

    def d_inf(o1, o2):
        return np.max(np.abs(o1 - o2))