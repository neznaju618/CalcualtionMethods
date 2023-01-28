import numpy as np
import matplotlib.pyplot as plt

from mnk.mnk_alg.MNKDataset import MNKDataset
from mnk.mnk_alg.Plotter import plot, plot_NN, plot_NN_poly
from neuralnetwork.ActivationFunctions import ActivationFunctions
from neuralnetwork.ErrorFunctions import ErrorFunctions
from neuralnetwork.LayerInfo import LayerInfo
from neuralnetwork.NeuralNetwork import NeuralNetwork

layers_info = [LayerInfo(ActivationFunctions.SIGMOID, 10),
               LayerInfo(ActivationFunctions.TANH, 10),
               LayerInfo(ActivationFunctions.LINEAR, 1)]
NN = NeuralNetwork(layers_info, 2)

#f = lambda x: 3/5*x + 7/5*x**3 + 7*np.sin(5*x)
f = lambda x: x ** 2 - x# + np.sin(3 * x)
a, b = -1, 1

dataset = MNKDataset(a, b, f)
dataset.initialize(15)

x = []
y = []
for i in range(len(dataset)):
    x.append(np.array([dataset.x()[i], dataset.x()[i]**2]))
    y.append(np.array([dataset.y()[i]]))

x_train = np.array(x)
y_train = np.array(y)

valid = MNKDataset(a, b, f)
valid.initialize(200)

x_valid = []
y_valid = []
for i in range(len(valid)):
    x_valid.append(np.array([valid.x()[i], valid.x()[i]**2]))
    y_valid.append(np.array([valid.y()[i]]))

x_valid = np.array(x)
y_valid = np.array(y)

NN.sample_fit_one_output_function(x_train, y_train, x_valid, y_valid, ErrorFunctions.MSE, h=0.03, epochs=10000)

test = MNKDataset(a, b, f)
test.initialize(300)

x_test = []
y_test = []
for i in range(len(test)):
    x_test.append(np.array([test.x()[i], test.x()[i]**2]))
    y_test.append(np.array([test.y()[i]]))

x_test = np.array(x)
y_test = np.array(y)

plot(f, a, b, "f")
plot_NN_poly(NN, a, b, 'NN_poly', 2)
plt.legend()
plt.show()
print('MSE=', NN.sample_MSE_one_output_function(x_test, y_test))
