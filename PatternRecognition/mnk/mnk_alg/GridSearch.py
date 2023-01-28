import numpy as np

from mnk.mnk_alg.PolynomialRegressor import PolynomialRegressor


class ParametersPair:

    def __init__(self, M, alpha):
        self.M = M
        self.alpha = alpha

    def copy(self):
        return ParametersPair(self.M, self.alpha)


class GridSearch:

    def __init__(self, parameters, cv=3):
        self.cv = cv
        self.parameters = parameters
        self.best_params = None
        self.best_estimator = None

    def fit(self, training_sample):
        score_results = {}
        subsamples = self.__split_training_sample(training_sample)

        for deg in self.parameters["M"]:
            for alpha in self.parameters["alpha"]:
                params = ParametersPair(deg, alpha)
                score_results[params] = self.__fit_estimator(subsamples, deg, alpha)

        self.best_params = self.__find_best_params(score_results)
        self.best_estimator = PolynomialRegressor(self.best_params.M, self.best_params.alpha)
        self.best_estimator.fit_by_MNK(training_sample.x(), training_sample.y())

    @staticmethod
    def __find_best_params(score_results):
        min_error = 10e100
        best_params_pair = None
        for key in score_results.keys():
            if score_results[key] < min_error:
                min_error = score_results[key]
                best_params_pair = key.copy()
        return best_params_pair

    def __split_training_sample(self, training_sample):
        subsamples = []
        length = len(training_sample)
        for i in range(self.cv):
            start_idx = length // self.cv * i
            finish_idx = length // self.cv * (i + 1)
            if i == self.cv - 1:
                finish_idx = length
            subsamples.append(training_sample.get_subset(start_idx, finish_idx))
        return subsamples

    @staticmethod
    def __fit_estimator(subsamples, M, alpha):
        estimator = PolynomialRegressor(M, alpha)
        scores = []
        for i in range(len(subsamples)):
            clone_est = estimator.copy()
            test = subsamples[i]
            train_subsets_x = []
            train_subsets_y = []
            for j in range(len(subsamples)):
                if i != j:
                    train_subsets_x.append(subsamples[j].x())
                    train_subsets_y.append(subsamples[j].y())
            train_x = np.concatenate(train_subsets_x)
            train_y = np.concatenate(train_subsets_y)
            clone_est.fit_by_MNK(train_x, train_y)
            scores.append(clone_est.get_MSE(test.x(), test.y()))
        return np.mean(scores)
