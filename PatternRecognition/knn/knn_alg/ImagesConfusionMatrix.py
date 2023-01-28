import numpy as np
import pandas as pd


# строки - действиетльные значения
# столбцы - предсказанные значения
class ImagesConfusionMatrix:

    def __init__(self, y, y_pred):
        self.__matrix = np.zeros((10, 10))
        for i in range(len(y)):
            self.__matrix[int(y[i]), int(y_pred[i])] += 1

    def show(self):
        result = pd.DataFrame(self.__matrix.copy() / np.sum(self.__matrix, axis=1))
        return result
