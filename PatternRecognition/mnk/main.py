import matplotlib.pyplot as plt
import numpy as np

from mnk.mnk_alg.GridSearch import GridSearch
from mnk.mnk_alg.MNKDataset import MNKDataset
from mnk.mnk_alg.Plotter import plot
from mnk.mnk_alg.PolynomialRegressor import PolynomialRegressor

f = lambda x: 5*x**3 + x**2 + 5
a, b = -1, 1

dataset = MNKDataset(a, b, f)
dataset.initialize(25)
test = dataset.generate_test_dataset(25)

reg_grid = GridSearch(parameters={"M": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "alpha": [0, 0.001, 0.01, 0.1, 1]},
                      cv=3)
reg_grid.fit(dataset)
reg = reg_grid.best_estimator
plt.scatter(dataset.x(), dataset.y(), label='test')
plot(f, a, b, "f")
plot(reg, a, b, 'mnk')
plt.legend()
plt.show()
print('Best parameters: M=', reg_grid.best_params.M, 'alpha=', reg_grid.best_params.alpha)
print('MSE=', reg.get_MSE(test.x(), test.y()))
