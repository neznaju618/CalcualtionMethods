import numpy as np
import matplotlib.pyplot as plt

from neuralnetwork.ActivationFunctions import ActivationFunctions
from neuralnetwork.Layer import Layer
from neuralnetwork.LayerInfo import LayerInfo


class NeuralNetwork:

    def __init__(self, layers_info, N_inputs):
        self.__layers = [Layer(LayerInfo(f_act=ActivationFunctions.LINEAR, N_neurons=N_inputs))]
        for i in range(len(layers_info)):
            self.__layers.append(Layer(layers_info[i]))
            self.__layers[i + 1].set_prev_layer(self.__layers[i])

    def __first_layer(self):
        return self.__layers[1]

    def __last_layer(self):
        return self.__layers[len(self.__layers) - 1]

    def output(self, x):
        return self.__last_layer().output(x)

    def fit(self, x, y_true, loss_function, h):
        y_pred = self.output(x)
        self.__first_layer().fit(y_pred, y_true, loss_function, h)

    def fit_by_dataset(self, x_train, y_train, loss_function, h, epochs):
        for i in range(epochs):
            print('epoch=', i)
            for j in range(len(x_train)):
                self.fit(x_train[j], y_train[j], loss_function, h)

    def sample_fit_one_output_function(self, x_train, y_train, x_valid, y_valid, loss_function, h, epochs):
        MSE_valid_list = []
        MSE_train_list = []
        epochs_list = []
        for i in range(epochs):
            for j in range(len(x_train)):
                self.fit(x_train[j], y_train[j], loss_function, h)
            epochs_list.append(i)
            MSE_valid_list.append(self.sample_MSE_one_output_function(x_valid, y_valid))
            MSE_train_list.append(self.sample_MSE_one_output_function(x_train, y_train))
            print('epoch=', i, 'train_MSE=', MSE_train_list[i], 'valid_MSE=', MSE_valid_list[i])
        plt.plot(epochs_list, MSE_train_list, label='Train MSE')
        plt.plot(epochs_list, MSE_valid_list, label='Valid MSE')
        plt.legend()
        plt.show()

    def sample_MSE_one_output_function(self, x, y_true):
        sum = 0
        for i in range(len(x)):
            sum += (self.output(x[i])[0] - y_true[i][0])**2
        return sum / len(x)

    def __call__(self, x):
        return self.output(x)
