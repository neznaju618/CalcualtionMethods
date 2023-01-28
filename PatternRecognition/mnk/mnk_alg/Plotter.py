import matplotlib.pyplot as plt
import numpy as np


def plot(f, a, b, name):
    N = 250
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = a
    y[0] = f(a)
    for i in range(1, N):
        x[i] = x[i - 1] + (b - a) / N
        y[i] = f(x[i])
    plt.plot(x, y, label=name)


def plot_NN(f, a, b, name):
    N = 10000
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = a
    y[0] = f(np.array([a]))[0]
    for i in range(1, N):
        x[i] = x[i - 1] + (b - a) /N
        y[i] = f(np.array([x[i]]))[0]
    plt.plot(x, y, label=name)


def plot_NN_poly(f, a, b, name, deg):
    N = 10000
    x = np.zeros(N)
    y = np.zeros(N)
    x_NN = []
    x[0] = a
    line1 = []
    for j in range(1, deg+1):
        line1.append(x[0]**j)
    x_NN.append(np.array(line1))
    y[0] = f(x_NN[0])[0]
    for i in range(1, N):
        x[i] = x[i - 1] + (b - a) /N
        line = []
        for j in range(1, deg + 1):
            line.append(x[i] ** j)
        x_NN.append(np.array(line))
        y[i] = f(x_NN[i])[0]
    plt.plot(x, y, label=name)