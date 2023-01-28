import numpy as np


class Layer:

    def __init__(self, layer_info):
        self.__weights = None
        self.__next_layer = None
        self.__prev_layer = None
        self.__f_act = layer_info.f_act
        self.__size = layer_info.N_neurons
        self.__b = np.random.normal(size=self.__size)
        self.__dfdx = None
        self.__dfdw = None
        self.__dfdb = None
        self.__output_before_f = None

    def set_prev_layer(self, prev_layer):
        self.__prev_layer = prev_layer
        prev_layer.__next_layer = self
        self.__weights = np.random.normal(loc=0, scale=0.001, size=(self.__size, prev_layer.__size))

    def output(self, x):
        if self.__prev_layer is None:
            return x
        prev_output = self.__prev_layer.output(x)
        self.__output_before_f = self.__weights.dot(prev_output) + self.__b
        self.__dfdb = self.__f_act.derivative(self.__output_before_f)
        f_der_out = np.array(self.__dfdb).reshape(-1, 1)
        self.__dfdx = f_der_out * self.__weights
        self.__dfdw = f_der_out.dot(np.array(prev_output).reshape(1, -1))
        return self.__f_act.function(self.__output_before_f)

    def fit(self, y_pred, y_true, loss_function, h):
        if self.__next_layer is None:
            dLdf = loss_function.derivative(y_pred, y_true)
            dLdx = dLdf.dot(self.__dfdx)
            cur_b_grad = dLdf * self.__dfdb
            cur_grad = np.array(dLdf).reshape(-1, 1) * self.__dfdw
            self.__weights -= h * cur_grad
            self.__b -= h * cur_b_grad
            return dLdx
        dLdf = self.__next_layer.fit(y_pred, y_true, loss_function, h)
        dLdx = dLdf.dot(self.__dfdx)
        cur_b_grad = dLdf * self.__dfdb
        cur_grad = np.array(dLdf).reshape(-1, 1) * self.__dfdw
        self.__b -= h * cur_b_grad
        self.__weights -= h * cur_grad
        return dLdx
