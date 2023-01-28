import numpy as np

from mnk.mnk_alg.DataPreprocessor import DataPreprocessor


class PolynomialRegressor:

    def __init__(self, M, alpha=0):
        self.__deg = M + 1
        self.__data_prep = DataPreprocessor(M)
        self.__weights = np.random.random(size=self.__deg)
        self.__alpha = alpha

    def predict(self, x):
        X = self.__data_prep.preprocess(x)
        return X.dot(self.__weights)

    def get_MSE(self, x, y):
        return 1 / len(y) * np.sum(np.square(y - self.predict(x)))

    def fit_by_MNK(self, train_x, train_y):
        X = self.__data_prep.preprocess(train_x)
        A = X.T.dot(X)
        b = X.T.dot(train_y)
        diag = np.eye(self.__deg) * self.__alpha
        diag[0, 0] = 0
        A += diag
        self.__weights = np.linalg.solve(A, b)

    def fit_by_GD(self, train_x, train_y, epochs, batch_size, h):
        X = self.__data_prep.preprocess(train_x)
        N, M = X.shape
        for k in range(epochs):
            for i in range(N // batch_size + 1):
                last_idx = (i + 1) * batch_size if (i + 1) * batch_size > N else N
                X_b = X[i * batch_size: last_idx]
                y_b = np.array(train_y[i * batch_size: last_idx])
                grad = (y_b - X_b.dot(self.__weights)).dot(X_b) / batch_size
                w = self.__weights
                w[0] = 0
                grad -= self.__alpha * w
                self.__weights += h * grad
            print("MSE=", self.get_MSE(train_x, train_y))

    def __call__(self, x):
        return self.predict(x)

    def copy(self):
        res = PolynomialRegressor(self.__deg-1, self.__alpha)
        res.__weights = self.__weights
        return res
